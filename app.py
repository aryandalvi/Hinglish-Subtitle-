"""
Hinglish Transcriber
====================

A small offline desktop app (tkinter) that turns an audio/video file into:
  * a Hinglish transcript  (<audio_name>_hinglish.txt)
  * phone-friendly Hinglish subtitles  (<audio_name>_hinglish.srt)

Pipeline
--------
1. faster-whisper transcribes the audio as Hindi (Devanagari) with word timestamps.
2. The Devanagari text is converted to WhatsApp-style Hinglish (Roman script), either
     a) offline: dictionaries + indic-transliteration + cleanup rules, or
     b) with the Claude API (better quality, needs internet + API key).
3. Words are grouped into short subtitle lines (max 3 words / 20 characters) and timed
   from the word timestamps. Each cue lasts at least 0.8 s and cues never overlap.

Low-confidence words/segments are marked with [?] so you know what to double-check.

Run:  python app.py
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import sys
import threading
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def add_nvidia_dll_dirs() -> None:
    """Make pip-installed cuBLAS/cuDNN DLLs (site-packages/nvidia/*/bin) findable on Windows."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_dirs()
# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

APP_TITLE = "Hinglish Transcriber"
MODEL_SIZES = ["small", "medium", "large-v3"]
DEFAULT_MODEL = "medium"
AUDIO_TYPES = [
    ("Audio / video files", "*.mp3 *.wav *.m4a *.ogg *.flac *.mp4"),
    ("All files", "*.*"),
]

# Subtitle rules (phone videos)
MAX_WORDS_PER_CUE = 3
SOFT_MIN_WORDS_PER_CUE = 2        # may break early at punctuation once this many words
MAX_CHARS_PER_CUE = 20
MIN_CUE_DURATION_MS = 800

# Style example for Whisper: Hindi in Devanagari, English words in English spelling.
# Whisper copies this style, so English words come out as "age", "question", not एज, क्वेश्चन.
# Keep it different from real video lines: Whisper skips audio that repeats the prompt.
WHISPER_PROMPT = ("नमस्ते दोस्तों, आज का topic है dental care. अगर आपके teeth में pain या "
                  "sensitivity है, तो checkup के लिए appointment ज़रूर लीजिए। Orthodontic "
                  "treatment, braces, aligners, implants, possible, healthy, supporting.")

# Confidence thresholds for the [?] markers
LOW_WORD_PROBABILITY = 0.50       # faster-whisper word.probability (0..1)
LOW_SEGMENT_LOGPROB = -1.0        # faster-whisper segment.avg_logprob
UNSURE = "[?]"

# Claude API mode
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_BATCH_SIZE = 40            # segments per request

# Optional user dictionary next to app.py / the .exe: lines like  "क्लिनिक = clinic"
CUSTOM_DICT_NAME = "hinglish_custom.txt"


def app_dir() -> Path:
    """Folder of app.py, or of the .exe when packaged with PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------

@dataclass
class Word:
    text: str           # word text (Devanagari from Whisper, or Hinglish after conversion)
    start: float        # seconds
    end: float          # seconds
    low: bool = False   # low confidence -> gets a [?] marker


@dataclass
class Segment:
    start: float
    end: float
    text: str                                         # Devanagari text from Whisper
    words: list[Word] = field(default_factory=list)   # Devanagari words with timestamps
    low: bool = False                                  # whole segment is low confidence
    hinglish: str = ""                                 # converted text (no markers)
    hinglish_words: list[Word] = field(default_factory=list)


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


# --------------------------------------------------------------------------------------
# Offline Hinglish conversion
# --------------------------------------------------------------------------------------

# Common Hindi words have a "conventional" chat spelling that rules can't guess well.
COMMON_HINDI = {
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the", "हो": "ho",
    "हूँ": "hoon", "हूं": "hoon", "में": "mein", "मैं": "main", "नहीं": "nahi", "नही": "nahi",
    "क्या": "kya", "क्यों": "kyun", "क्यूँ": "kyun", "क्यूं": "kyun", "कैसे": "kaise",
    "कैसा": "kaisa", "कैसी": "kaisi", "तो": "toh", "यह": "yeh", "ये": "ye", "वह": "woh",
    "वो": "woh", "और": "aur", "भी": "bhi", "कि": "ki", "की": "ki", "का": "ka", "के": "ke",
    "को": "ko", "से": "se", "पर": "par", "ने": "ne", "जो": "jo", "एक": "ek", "दो": "do",
    "तीन": "teen", "चार": "chaar", "पांच": "paanch", "पाँच": "paanch", "कुछ": "kuch",
    "सब": "sab", "सबको": "sabko", "बहुत": "bahut", "अब": "ab", "अभी": "abhi", "जब": "jab",
    "तब": "tab", "कब": "kab", "यहाँ": "yahan", "यहां": "yahan", "वहाँ": "wahan",
    "वहां": "wahan", "कहाँ": "kahan", "कहां": "kahan", "हाँ": "haan", "हां": "haan",
    "ना": "na", "न": "na", "मुझे": "mujhe", "मुझको": "mujhko", "तुम": "tum", "तुम्हें": "tumhe",
    "तुम्हारा": "tumhara", "तुम्हारी": "tumhari", "आप": "aap", "आपका": "aapka",
    "आपकी": "aapki", "आपके": "aapke", "आपको": "aapko", "हम": "hum", "हमें": "humein",
    "हमारा": "humara", "हमारी": "humari", "हमारे": "humare", "उनका": "unka", "उनकी": "unki",
    "उसका": "uska", "उसकी": "uski", "उसके": "uske", "उसे": "use", "उन्हें": "unhe",
    "इस": "is", "उस": "us", "इसे": "ise", "इसका": "iska", "इसकी": "iski", "इसके": "iske",
    "इसलिए": "isliye", "लेकिन": "lekin", "मगर": "magar", "या": "ya", "अगर": "agar",
    "फिर": "phir", "बस": "bas", "ही": "hi", "सिर्फ": "sirf", "सिर्फ़": "sirf",
    "ज्यादा": "zyada", "ज़्यादा": "zyada", "थोड़ा": "thoda", "थोड़ी": "thodi", "अच्छा": "accha",
    "अच्छी": "acchi", "अच्छे": "acche", "ठीक": "theek", "सही": "sahi", "बात": "baat",
    "बातें": "baatein", "लोग": "log", "लोगों": "logon", "दिन": "din", "रात": "raat",
    "समय": "samay", "वक्त": "waqt", "वक़्त": "waqt", "साल": "saal", "कोई": "koi",
    "किसी": "kisi", "कौन": "kaun", "कहा": "kaha", "कहते": "kehte", "कहना": "kehna",
    "कहती": "kehti", "रहा": "raha", "रही": "rahi", "रहे": "rahe", "गया": "gaya", "गई": "gayi",
    "गयी": "gayi", "गए": "gaye", "गये": "gaye", "दिया": "diya", "लिया": "liya", "किया": "kiya",
    "करो": "karo", "करना": "karna", "करें": "karein", "करते": "karte", "करता": "karta",
    "करती": "karti", "होता": "hota", "होती": "hoti", "होते": "hote", "होगा": "hoga",
    "होगी": "hogi", "होंगे": "honge", "जाता": "jaata", "जाती": "jaati", "जाते": "jaate",
    "जाओ": "jao", "आओ": "aao", "चलो": "chalo", "देखो": "dekho", "सुनो": "suno",
    "मतलब": "matlab", "यार": "yaar", "भाई": "bhai", "जी": "ji", "वाला": "wala",
    "वाली": "wali", "वाले": "wale", "पहले": "pehle", "बाद": "baad", "साथ": "saath",
    "लिए": "liye", "लिये": "liye", "तरह": "tarah", "ऐसा": "aisa", "ऐसे": "aise",
    "ऐसी": "aisi", "वैसे": "waise", "जैसे": "jaise", "क्योंकि": "kyunki", "मेरा": "mera",
    "मेरी": "meri", "मेरे": "mere", "तेरा": "tera", "तेरी": "teri", "अपना": "apna",
    "अपनी": "apni", "अपने": "apne", "सकते": "sakte", "सकता": "sakta", "सकती": "sakti",
    "चाहिए": "chahiye", "पता": "pata", "कहीं": "kahin", "नमस्ते": "namaste",
    "नमस्कार": "namaskar", "धन्यवाद": "dhanyavaad", "शुक्रिया": "shukriya",
    "ज़िंदगी": "zindagi", "जिंदगी": "zindagi", "दोस्त": "dost", "दोस्तों": "doston",
    "घर": "ghar", "पानी": "paani", "खाना": "khana", "बच्चा": "bachcha", "बच्चे": "bachche",
    "बच्चों": "bachchon", "ज़रूर": "zaroor", "जरूर": "zaroor", "ज़रूरी": "zaroori",
    "जरूरी": "zaroori", "शायद": "shayad", "हमेशा": "hamesha", "कभी": "kabhi",
    "आज": "aaj", "कल": "kal", "परसों": "parson", "चीज़": "cheez", "चीज": "cheez",
    "चीज़ें": "cheezein", "चीजें": "cheezein", "वजह": "wajah", "मैंने": "maine",
    "उसने": "usne", "हमने": "humne", "आपने": "aapne", "उन्होंने": "unhone", "ऊपर": "upar",
    "नीचे": "neeche", "अंदर": "andar", "बाहर": "bahar", "सच": "sach", "प्यार": "pyaar",
    "दिल": "dil", "खुश": "khush", "ख़ुश": "khush", "परेशान": "pareshan",
    "परेशानी": "pareshani", "दर्द": "dard", "दांत": "daant", "दाँत": "daant",
    "दांतों": "daanton", "शुरू": "shuru", "ख़त्म": "khatam", "खत्म": "khatam",
    "फायदा": "faayda", "फ़ायदा": "faayda", "फाइदा": "faayda", "कीजिए": "kijiye",
    "कीजिये": "kijiye", "कीजे": "kijiye", "वज़े": "wajah", "मिलता": "milta", "सुनने": "sunne", "खतम": "khatam",
    "बिलता": "milta",
}

# English words that Whisper writes phonetically in Devanagari -> normal English spelling.
ENGLISH_LOANWORDS = {
    "फ़ोन": "phone", "पासवर्ड": "password", "टेंशन": "tension", "स्माइल": "smile",
    "क्लिनिक": "clinic", "क्लिनिक्स": "clinics", "सिक्योरिटी": "security",
    "सिक्यूरिटी": "security", "डॉक्टर": "doctor", "डाक्टर": "doctor", "डॉक्टर्स": "doctors",
    "मोबाइल": "mobile", "इंटरनेट": "internet", "वीडियो": "video", "वीडियोज़": "videos",
    "वीडियोस": "videos", "चैनल": "channel", "सब्सक्राइब": "subscribe", "लाइक": "like",
    "शेयर": "share", "कमेंट": "comment", "कमेंट्स": "comments", "ऑफिस": "office",
    "ऑफ़िस": "office", "मीटिंग": "meeting", "प्रॉब्लम": "problem", "प्रोब्लम": "problem",
    "प्रॉब्लम्स": "problems", "सॉल्यूशन": "solution", "टाइम": "time", "प्लान": "plan",
    "प्लीज़": "please", "प्लीज": "please", "सॉरी": "sorry", "थैंक": "thank",
    "थैंक्स": "thanks", "थैंक्यू": "thank you", "ओके": "okay", "हेलो": "hello",
    "हैलो": "hello", "बाय": "bye", "स्कूल": "school", "कॉलेज": "college", "क्लास": "class",
    "टीचर": "teacher", "स्टूडेंट": "student", "स्टूडेंट्स": "students", "एग्जाम": "exam",
    "एग्ज़ाम": "exam", "जॉब": "job", "कंपनी": "company", "बिज़नेस": "business",
    "बिजनेस": "business", "मार्केट": "market", "मार्केटिंग": "marketing",
    "प्रोडक्ट": "product", "प्रोडक्ट्स": "products", "सर्विस": "service",
    "कस्टमर": "customer", "कस्टमर्स": "customers", "पेमेंट": "payment", "बैंक": "bank",
    "अकाउंट": "account", "कार्ड": "card", "ऑनलाइन": "online", "ऑफलाइन": "offline",
    "ऐप": "app", "एप": "app", "ऐप्स": "apps", "वेबसाइट": "website", "लिंक": "link",
    "डिस्क्रिप्शन": "description", "कंप्यूटर": "computer", "लैपटॉप": "laptop",
    "सिस्टम": "system", "डेटा": "data", "डाटा": "data", "सॉफ्टवेयर": "software",
    "हॉस्पिटल": "hospital", "मेडिसिन": "medicine", "हेल्थ": "health", "फिटनेस": "fitness",
    "जिम": "gym", "डाइट": "diet", "टेस्ट": "test", "रिपोर्ट": "report", "रिजल्ट": "result",
    "ट्रीटमेंट": "treatment", "पेशेंट": "patient", "पेशेंट्स": "patients",
    "डेंटिस्ट": "dentist", "डेंटल": "dental", "टीथ": "teeth", "स्किन": "skin",
    "हेयर": "hair", "फेस": "face", "केयर": "care", "ब्रांड": "brand", "प्राइस": "price",
    "ऑफर": "offer", "डिस्काउंट": "discount", "फ्री": "free", "शॉपिंग": "shopping",
    "ऑर्डर": "order", "डिलीवरी": "delivery", "कार": "car", "ट्रेन": "train",
    "टिकट": "ticket", "होटल": "hotel", "रूम": "room", "फैमिली": "family",
    "फ्रेंड": "friend", "फ्रेंड्स": "friends", "पार्टी": "party", "म्यूजिक": "music",
    "म्यूज़िक": "music", "सॉन्ग": "song", "मूवी": "movie", "फिल्म": "film", "लाइफ": "life",
    "लव": "love", "हैप्पी": "happy", "स्टोरी": "story", "सीक्रेट": "secret", "टिप": "tip",
    "टिप्स": "tips", "ट्रिक": "trick", "ट्रिक्स": "tricks", "आइडिया": "idea",
    "स्टेप": "step", "स्टेप्स": "steps", "बेस्ट": "best", "गुड": "good", "बैड": "bad",
    "नॉर्मल": "normal", "सिंपल": "simple", "एक्चुअली": "actually", "बेसिकली": "basically",
    "सीरियसली": "seriously", "रियली": "really", "एक्सपीरियंस": "experience",
    "इंपॉर्टेंट": "important", "इम्पोर्टेन्ट": "important", "कॉन्फिडेंस": "confidence",
    "मोटिवेशन": "motivation", "स्ट्रेस": "stress", "प्रेशर": "pressure", "लेवल": "level",
    "पॉइंट": "point", "नंबर": "number", "मैसेज": "message", "कॉल": "call", "ईमेल": "email",
    "सेटिंग": "setting", "सेटिंग्स": "settings", "अपडेट": "update", "डाउनलोड": "download",
    "अपलोड": "upload", "चार्ज": "charge", "बैटरी": "battery", "स्क्रीन": "screen",
    "कैमरा": "camera", "फोटो": "photo", "फ़ोटो": "photo", "सेल्फी": "selfie", "पोस्ट": "post",
    "इंस्टाग्राम": "Instagram", "यूट्यूब": "YouTube", "व्हाट्सएप": "WhatsApp",
    "व्हाट्सऐप": "WhatsApp", "फेसबुक": "Facebook", "गूगल": "Google", "सर": "sir",
    "मैडम": "madam", "मैम": "ma'am", "बेबी": "baby", "क्लीन": "clean", "क्लीनिंग": "cleaning",
    "ब्रश": "brush", "पेस्ट": "paste", "ट्रैफिक": "traffic", "पुलिस": "police",
    "पॉलिसी": "policy", "इंश्योरेंस": "insurance", "लोन": "loan",
    "इन्वेस्टमेंट": "investment", "मनी": "money", "सेविंग": "saving", "सेविंग्स": "savings",
    "बजट": "budget", "टैक्स": "tax", "सेफ": "safe", "सेफ्टी": "safety", "अलर्ट": "alert",
    "रिस्क": "risk", "फ्रॉड": "fraud", "स्कैम": "scam", "हैक": "hack", "हैकर": "hacker",
    "हैकर्स": "hackers", "ओटीपी": "OTP", "पिन": "PIN", "एरर": "error", "इश्यू": "issue",
    "चेक": "check", "चेंज": "change", "स्टार्ट": "start", "स्टॉप": "stop", "रेडी": "ready",
    "सपोर्ट": "support", "हेल्प": "help", "टीम": "team", "मैनेजर": "manager", "बॉस": "boss",
    "सैलरी": "salary", "प्रोजेक्ट": "project", "डेडलाइन": "deadline", "एनर्जी": "energy",
    "वॉटर": "water", "ड्रिंक": "drink", "फूड": "food", "फ़ूड": "food", "हेल्दी": "healthy",
    "वीक": "week", "मंथ": "month", "मॉर्निंग": "morning", "नाइट": "night",
    "वीकेंड": "weekend", "एक्सरसाइज": "exercise", "एक्सरसाइज़": "exercise", "योगा": "yoga",
    "ब्यूटी": "beauty", "मेकअप": "makeup", "ड्रेस": "dress", "फैशन": "fashion",
    "स्टाइल": "style", "लुक": "look", "स्पेशल": "special", "ऑप्शन": "option",
    "ऑप्शंस": "options", "क्वालिटी": "quality", "टॉपिक": "topic", "कंटेंट": "content",
    "क्रिएटर": "creator", "फॉलो": "follow", "फॉलोवर्स": "followers", "ब्लॉग": "blog",
    "व्लॉग": "vlog", "सेशन": "session", "कोर्स": "course", "ट्रेनिंग": "training",
    "स्किल": "skill", "स्किल्स": "skills", "करियर": "career", "इंटरव्यू": "interview",
    "सिटी": "city", "एरिया": "area", "लोकेशन": "location", "एड्रेस": "address",
    "पार्किंग": "parking", "पेट्रोल": "petrol", "बाइक": "bike", "ड्राइवर": "driver",
    "मास्क": "mask", "वायरस": "virus", "वैक्सीन": "vaccine", "इम्प्लांट": "implant",
    "इंप्लांट": "implant", "रूट": "root", "कैनाल": "canal", "ब्रेसेस": "braces",
    "ऑपरेशन": "operation", "सर्जरी": "surgery", "अपॉइंटमेंट": "appointment",
    "कंसल्टेशन": "consultation", "ट्रस्ट": "trust", "रिव्यू": "review",
    "रिव्यूज": "reviews", "रेटिंग": "rating", "सब्सक्राइबर्स": "subscribers", "ऑफ": "off",
    "ऑन": "on", "यस": "yes", "नो": "no", "वाओ": "wow", "ओह": "oh", "सो": "so",
    "बट": "but", "एंड": "and", "एक्स्ट्रा": "extra", "फ़ास्ट": "fast", "फास्ट": "fast",
    "एज": "age", "स्ट्रेट": "straight", "क्वेश्चन": "question", "क्वेशचन": "question",
    "क्वेश्चंस": "questions", "फ्रीक्वेंटली": "frequently", "आंसर": "answer", "आन्सर": "answer",
    "एक्सपायरी": "expiry", "एक्सपैरी": "expiry", "एक्सापैरी": "expiry", "डेट": "date",
    "डिसाइड": "decide", "डिसाइट": "decide", "डिसैट": "decide", "एडल्ट्स": "adults",
    "अडल्ट्स": "adults", "अडल्स": "adults", "अदल्स": "adults", "ऑर्थोडॉन्टिक": "orthodontic",
    "ऑर्थोडोंटिक": "orthodontic", "ओथ्रोंटिक": "orthodontic", "ऑर्थोडॉन्टिक्स": "orthodontics",
    "ऑर्थोडोंटिक्स": "orthodontics", "ओथ्रोंटिक्स": "orthodontics", "ट्रीटमेन्ट": "treatment",
    "पॉसिबल": "possible", "प्रोवाइडेड": "provided", "प्रोवाइडिट": "provided", "गम्स": "gums",
    "सपोर्टिंग": "supporting", "बोन": "bone", "बून": "bone", "इनविज़लाइन": "Invisalign",
    "इनविजलाइन": "Invisalign", "क्लियर": "clear", "अलाइनर": "aligner", "अलाइनर्स": "aligners",
    "डिस्क्रीट": "discreet", "डिसक्रीट": "discreet", "डिसक्रिट": "discreet",
    "डिसाक्रिट": "discreet", "सोशल": "social", "कंटिन्यू": "continue", "कन्टिन्यू": "continue",
    "डोक्रे": "Dr.", "डॉ": "Dr.", "इयर्स": "years", "येर्स": "years", "ईयर्स": "years",
    "अगो": "ago", "ब्रेसिस": "braces", "अवॉइड": "avoid", "ऑपर्चुनिटी": "opportunity",
    "ऑपरचुनिटी": "opportunity", "अपॉर्चुनिटी": "opportunity", "सूटेबिलिटी": "suitability",
    "सूटेबलिटी": "suitability", "डायग्नोसिस": "diagnosis", "डाइग्नोसिस": "diagnosis",
    "कंसिलिटेशन": "consultation", "कन्सिलिटेशन": "consultation", "कंसल्टेशन्स": "consultations",
    "शेड्यूल": "schedule", "शेडूल": "schedule", "शेड्युल": "schedule", "हेल्दी": "healthy",
    "एमडीएस": "MDS", "बीडीएस": "BDS", "वाशी": "Vashi", "बाशी": "Vashi", "बाशिय": "Vashi",
    "नेरूल": "Nerul", "स्माइल्स": "smiles", "अलाइनमेंट": "alignment", "क्राउन": "crown",
    "फिलिंग": "filling", "कैविटी": "cavity", "एक्स-रे": "X-ray", "स्कैन": "scan",
    "पूस्सूबल": "possible", "पॉस्टूबल": "possible", "पॉसिबल्": "possible",
    "प्रॉवाइड़ेट": "provided", "प्रॉवाइडट": "provided", "सपोटिंग": "supporting",
    "हैल्डी": "healthy", "हेल्डी": "healthy", "बाशिये": "Vashi", "बाशिया": "Vashi",
    "क्वेश्टन": "question", "फ्रेक्वेंटली": "frequently", "अंसर": "answer", "सिमपल": "simple",
    "एक्स्पारी": "expiry", "आडल्स": "adults", "आउठरॉन्टिक": "orthodontic",
    "अउठरॉन्टिक": "orthodontic", "आउठरॉन्टिक्स": "orthodontics", "टीट": "teeth",
    "इन्विसलाइन": "Invisalign", "इलायनर्स": "aligners", "स्मायल": "smile", "प्लीस": "please",
    "दोक्टर": "doctor", "दॉक्टर": "doctor", "प्रवाइड़": "provided", "प्रवाइडेड": "provided",
    "वाशिया": "Vashi", "शढूल": "schedule",
}

VIRAMA = "्"
NUKTA = "़"
NASAL_SIGNS = {"ँ", "ं"}   # chandrabindu, anusvara
VISARGA = "ः"
PUNCT = set("।॥.,!?;:\"'“”‘’()[]{}…-–—")
SENTENCE_END = {"।": ".", "॥": "."}

# Characters Whisper sometimes uses that ITRANS doesn't know (English "o"/"e" sounds).
DEVANAGARI_FIXUPS = str.maketrans({"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े", "ॆ": "े", "ॊ": "ो"})


def _has_devanagari(s: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in s)


def _lookup_key(word: str) -> str:
    """Normalised key so 'फ़ोन' / 'फोन' / 'फॉन' style spelling variants share one entry."""
    w = unicodedata.normalize("NFC", word).translate(DEVANAGARI_FIXUPS)
    return w.replace(NUKTA, "").replace("ँ", "ं")


def _is_consonant(ch: str) -> bool:
    o = ord(ch)
    return 0x0915 <= o <= 0x0939 or 0x0958 <= o <= 0x095F or o in (0x0979, 0x097A)


def _is_matra(ch: str) -> bool:
    o = ord(ch)
    return 0x093E <= o <= 0x094C or o in (0x0962, 0x0963)


def _is_independent_vowel(ch: str) -> bool:
    o = ord(ch)
    return 0x0904 <= o <= 0x0914 or o in (0x0960, 0x0961, 0x0972)


def apply_schwa_deletion(word: str) -> str:
    """
    Hindi drops many of the "inherent a" sounds that Devanagari implies
    (करना is "karna", not "karanaa"). We add a virama (्) where the 'a' is silent so the
    transliterator doesn't print it. Standard rules, applied right-to-left:
      * the final 'a' of a word is silent  (घर -> ghar)
      * an 'a' between a vowel and a consonant+vowel is silent  (समझना -> samajhna)
    """
    units = []   # one entry per akshara (syllable unit)
    i = 0
    while i < len(word):
        ch = word[i]
        u = {"text": ch, "cons": False, "schwa": False, "vowel": False, "nasal": False}
        i += 1
        if _is_consonant(ch):
            u["cons"] = True
            if i < len(word) and word[i] == NUKTA:
                u["text"] += NUKTA
                i += 1
            if i < len(word) and word[i] == VIRAMA:          # half consonant (conjunct)
                u["text"] += VIRAMA
                i += 1
            elif i < len(word) and _is_matra(word[i]):       # consonant + vowel sign
                u["text"] += word[i]
                u["vowel"] = True
                i += 1
            else:                                            # consonant + inherent 'a'
                u["schwa"] = u["vowel"] = True
        elif _is_independent_vowel(ch):
            u["vowel"] = True
        while i < len(word) and (word[i] in NASAL_SIGNS or word[i] == VISARGA):
            u["text"] += word[i]
            u["nasal"] = True
            i += 1
        units.append(u)

    def silence(u):
        u["text"] += VIRAMA
        u["schwa"] = u["vowel"] = False

    n = len(units)
    if n < 2:
        return word
    last = units[-1]
    if last["schwa"] and not last["nasal"]:
        # Keep the 'a' after clusters ending in y/r/v: सूर्य surya, मित्र mitra.
        cluster_keep = units[-2]["cons"] and not units[-2]["vowel"] and last["text"][0] in "यरव"
        if not cluster_keep:
            silence(last)
    for k in range(n - 2, 0, -1):
        u, prev, nxt = units[k], units[k - 1], units[k + 1]
        if u["schwa"] and not u["nasal"] and prev["vowel"] and nxt["cons"] and nxt["vowel"]:
            silence(u)
    return "".join(u["text"] for u in units)


# ITRANS output -> (kind, Hinglish) tokens. kind: C=consonant, V=vowel, N=nasal, X=other.
_ITRANS_TOKENS = {
    "kSh": ("C", "ksh"), "j~n": ("C", "gy"), ".Dh": ("C", "dh"), ".D": ("C", "d"),
    "RRi": ("R", ""), "RRI": ("R", ""), "~N": ("C", "n"), "~n": ("C", "n"),
    ".N": ("N", ""), ".a": ("X", ""), "OM": ("X", "om"), "||": ("X", "."),
    "kh": ("C", "kh"), "gh": ("C", "gh"), "ch": ("C", "ch"), "Ch": ("C", "chh"),
    "jh": ("C", "jh"), "Th": ("C", "th"), "Dh": ("C", "dh"), "th": ("C", "th"),
    "dh": ("C", "dh"), "ph": ("C", "ph"), "bh": ("C", "bh"), "sh": ("C", "sh"),
    "Sh": ("C", "sh"), "aa": ("V", "aa"), "ii": ("V", "ii"), "uu": ("V", "uu"),
    "ai": ("V", "ai"), "au": ("V", "au"), "A": ("V", "aa"), "I": ("V", "ii"),
    "U": ("V", "uu"), "a": ("V", "a"), "i": ("V", "i"), "u": ("V", "u"), "e": ("V", "e"),
    "o": ("V", "o"), "M": ("N", ""), "H": ("X", ""), "T": ("C", "t"), "D": ("C", "d"),
    "N": ("C", "n"), "K": ("C", "kh"), "G": ("C", "gh"), "q": ("C", "k"), "Y": ("C", "y"),
    "|": ("X", "."),
}
for _c in "kgjtdnpbmyrlvszfh":
    _ITRANS_TOKENS.setdefault(_c, ("C", _c))
_ITRANS_KEYS = sorted(_ITRANS_TOKENS, key=len, reverse=True)


def _tokenize_itrans(s: str) -> list[tuple[str, str]]:
    tokens, i = [], 0
    while i < len(s):
        for key in _ITRANS_KEYS:
            if s.startswith(key, i):
                kind, val = _ITRANS_TOKENS[key]
                if kind == "R":                        # vocalic r (कृपया) -> "ri"
                    tokens += [("C", "r"), ("V", "i")]
                else:
                    tokens.append((kind, val))
                i += len(key)
                break
        else:
            tokens.append(("X", s[i]))                 # digits, Latin, anything unknown
            i += 1
    return tokens


def _render_tokens(tokens: list[tuple[str, str]]) -> str:
    """Turn ITRANS tokens into casual chat spelling (aa -> a where natural, ii -> ee/i, ...)."""
    vowel_idx = [i for i, (k, _) in enumerate(tokens) if k == "V"]
    mono = len(vowel_idx) == 1
    out = []
    for i, (kind, val) in enumerate(tokens):
        if kind == "V":
            first = vowel_idx[0] == i
            rest = tokens[i + 1:]
            final = all(k in ("N", "X") for k, _ in rest)        # last sound of the word
            nasal_next = bool(rest) and rest[0][0] == "N"
            if val == "aa":
                if final:
                    val = "aa" if mono and (i == 0 or nasal_next) else "a"   # aa, haan / ka, jana
                elif not first:
                    val = "a"                                  # bataana -> batana
            elif val == "ii":
                val = "i" if (final or not first) else "ee"    # theek, jaati, nahin
            elif val == "uu":
                val = "u" if final and not nasal_next else "oo"  # bhool, hoon, tu
            elif val == "e" and final and nasal_next:
                val = "ei"                                     # baatein, karein
            out.append(val)
        elif kind == "N":
            nxt = next((v for k, v in tokens[i + 1:] if k == "C"), "")
            out.append("m" if nxt[:1] in ("p", "b", "m") else "n")
        else:
            out.append(val)
    return "".join(out)


class OfflineConverter:
    """Devanagari -> Hinglish without internet: dictionaries first, then transliteration rules."""

    def __init__(self):
        from indic_transliteration import sanscript  # imported here so startup stays fast
        self._sanscript = sanscript
        self.dictionary: dict[str, str] = {}
        for table in (COMMON_HINDI, ENGLISH_LOANWORDS):
            for dev, roman in table.items():
                self.dictionary[_lookup_key(dev)] = roman
        self._load_custom_dictionary()

    def _load_custom_dictionary(self):
        path = app_dir() / CUSTOM_DICT_NAME
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                dev, roman = line.split("=", 1)
                if dev.strip() and roman.strip():
                    self.dictionary[_lookup_key(dev.strip())] = roman.strip()

    def word(self, word: str) -> str:
        """Convert one Devanagari word (no surrounding punctuation)."""
        if not _has_devanagari(word):
            if word.isupper() and len(word) > 4:                # Whisper's random CAPS -> lower
                return word.lower()
            return "I" if word == "i" else word                 # already Latin / digits
        hit = self.dictionary.get(_lookup_key(word))
        if hit:
            return hit
        dev = unicodedata.normalize("NFC", word).translate(DEVANAGARI_FIXUPS)
        dev = apply_schwa_deletion(dev)
        itrans = self._sanscript.transliterate(dev, self._sanscript.DEVANAGARI,
                                               self._sanscript.ITRANS)
        roman = _render_tokens(_tokenize_itrans(itrans))
        roman = unicodedata.normalize("NFKD", roman)           # strip any diacritics left
        return "".join(c for c in roman if not unicodedata.combining(c)).lower()

    def token(self, raw: str) -> str:
        """Convert a whitespace-separated token, keeping its punctuation (। -> .)."""
        lead, core, trail = split_punct(raw)
        return map_punct(lead) + self.word(core) + map_punct(trail)

    def text(self, text: str) -> str:
        return " ".join(self.token(t) for t in text.split())


def split_punct(raw: str) -> tuple[str, str, str]:
    a, b = 0, len(raw)
    while a < b and raw[a] in PUNCT:
        a += 1
    while b > a and raw[b - 1] in PUNCT:
        b -= 1
    return raw[:a], raw[a:b], raw[b:]


def map_punct(p: str) -> str:
    return "".join(SENTENCE_END.get(ch, ch) for ch in p)


def capitalize_sentences(words: list["Word"]) -> None:
    """Capitalise the first word of each sentence in place; leave the rest alone."""
    cap = True
    for w in words:
        if cap:
            for j, ch in enumerate(w.text):
                if ch.isalpha():
                    w.text = w.text[:j] + ch.upper() + w.text[j + 1:]
                    break
        cap = w.text.rstrip("\"')”’")[-1:] in (".", "?", "!")


# --------------------------------------------------------------------------------------
# Claude API conversion
# --------------------------------------------------------------------------------------

class ClaudeError(Exception):
    """Raised with a user-friendly message when the Claude API can't be used."""


CLAUDE_SYSTEM_PROMPT = """You convert Hindi speech transcripts written in Devanagari into natural Hinglish: Hindi written in Roman (Latin) script, exactly the way people in India type on WhatsApp.

Rules:
- Use casual, common Roman spellings for Hindi words: hai, nahi, kya, toh, mein, bhi, accha, theek, zindagi, jaati. No diacritics, no ITRANS/IAST notation, no capital letters inside words.
- Words that were spoken in English (Whisper often writes them phonetically in Devanagari, e.g. स्माइल, क्लिनिक, सिक्योरिटी, डॉक्टर, पासवर्ड) must be written in normal English spelling: smile, clinic, security, doctor, password.
- Do not translate, summarise, add or drop anything. Keep the word order, numbers and punctuation; write । as a full stop.
- Start each line with a capital letter. Everything else is lowercase except proper nouns, brand names and acronyms.

Input: one segment per line, formatted as <index><TAB><devanagari text>.
Output: exactly one line per input line, same order, formatted as <index><TAB><hinglish text>. Output nothing else: no commentary, no code fences.

Example input:
0	फ़ोन का पासवर्ड भूल जाओ तो टेंशन हो जाती है।
Example output:
0	Phone ka password bhool jao toh tension ho jaati hai."""

_LINE_RE = re.compile(r"^\s*(\d+)\s*[\t:|.)\-]\s*(.*?)\s*$")


class ClaudeConverter:
    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError as e:
            raise ClaudeError("The 'anthropic' package is not installed.\n\n"
                              "Run:  pip install anthropic") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=300)

    def convert(self, texts: dict[int, str]) -> dict[int, str]:
        """Convert {index: devanagari} in one request. Returns {index: hinglish} for the lines it got."""
        anthropic = self._anthropic
        body = "\n".join(f"{i}\t{t}" for i, t in texts.items())
        try:
            response = self.client.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=16000,
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": body}],
                output_config={"effort": "medium"},
                # If this model declines a request, the API retries it on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise ClaudeError("The Anthropic API key was rejected. Check the key and try again.") from e
        except anthropic.PermissionDeniedError as e:
            raise ClaudeError("This API key doesn't have access to the model.") from e
        except anthropic.RateLimitError as e:
            raise ClaudeError("Anthropic API rate limit reached. Wait a minute and try again.") from e
        except anthropic.APIStatusError as e:
            raise ClaudeError(f"Anthropic API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ClaudeError("Couldn't reach the Anthropic API. Check your internet connection.") from e

        if response.stop_reason == "refusal":
            return {}                                     # caller falls back to offline mode
        reply = "".join(b.text for b in response.content if b.type == "text")
        result = {}
        for line in reply.splitlines():
            m = _LINE_RE.match(line)
            if m and int(m.group(1)) in texts and m.group(2):
                result[int(m.group(1))] = m.group(2)
        return result


# --------------------------------------------------------------------------------------
# Aligning Hinglish words to timestamps
# --------------------------------------------------------------------------------------

_INITIALS_RE = re.compile(r"(?:[A-Z]\.)+")


def merge_punctuation(words: list[Word]) -> list[Word]:
    """
    Whisper sometimes returns ',' or '.' as a separate word, or splits "Dr.Sharad" into
    "Dr" + ".Sharad" and "M.D.S." into "M" + ".D" + ".S.". Glue punctuation to the word
    before and rejoin initials, so we get "Dr. Sharad" and "M.D.S.".
    """
    merged: list[Word] = []
    for w in words:
        text = w.text.strip()
        if merged and text.startswith(".") and not text.startswith(".."):
            merged[-1].text += "."
            text = text[1:].strip()
        if not text:
            continue
        if merged and all(ch in PUNCT for ch in text):
            merged[-1].text += text
            merged[-1].end = max(merged[-1].end, w.end)
        elif (merged and _INITIALS_RE.fullmatch(merged[-1].text)
              and _INITIALS_RE.match(text) and len(_INITIALS_RE.match(text).group()) == 2):
            merged[-1].text += text                      # "M.D." + "S." -> "M.D.S."
            merged[-1].end = max(merged[-1].end, w.end)
        else:
            merged.append(Word(text, w.start, w.end, w.low))
    return merged


def proportional_words(text: str, start: float, end: float) -> list[Word]:
    """Fallback when there are no word timestamps: share the segment time by text length."""
    tokens = text.split()
    total = sum(len(t) for t in tokens) or 1
    words, t = [], start
    for tok in tokens:
        dur = (end - start) * len(tok) / total
        words.append(Word(tok, t, t + dur))
        t += dur
    return words


def align_by_characters(hinglish: str, dev_words: list[Word], seg: Segment) -> list[Word]:
    """
    Give each Hinglish word a time span when the word count differs from Whisper's words
    (Claude mode). Position in the Hinglish line (by characters) is mapped onto the same
    relative position in the Devanagari words, then interpolated inside that word's timing.
    """
    tokens = hinglish.split()
    if not dev_words:
        return proportional_words(hinglish, seg.start, seg.end)

    dev_lens = [max(1, len(w.text)) for w in dev_words]
    dev_total = sum(dev_lens)
    bounds, acc = [0.0], 0
    for n in dev_lens:
        acc += n
        bounds.append(acc / dev_total)

    def time_at(f: float) -> tuple[float, int]:
        for k, w in enumerate(dev_words):
            if f <= bounds[k + 1] or k == len(dev_words) - 1:
                span = (bounds[k + 1] - bounds[k]) or 1
                inside = min(1.0, max(0.0, (f - bounds[k]) / span))
                return w.start + inside * (w.end - w.start), k
        return dev_words[-1].end, len(dev_words) - 1

    total = sum(len(t) for t in tokens) or 1
    words, acc = [], 0
    for tok in tokens:
        f0, f1 = acc / total, (acc + len(tok)) / total
        acc += len(tok)
        t0, _ = time_at(f0)
        t1, _ = time_at(f1)
        _, mid = time_at((f0 + f1) / 2)
        words.append(Word(tok, t0, max(t1, t0), low=dev_words[mid].low))
    return words


# --------------------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------------------

def _word_label(w: Word) -> str:
    return w.text + UNSURE if w.low else w.text


def chunk_words(words: list[Word], first_prefix: int = 0) -> list[list[Word]]:
    """
    Group words into lines of <= MAX_WORDS_PER_CUE words and <= MAX_CHARS_PER_CUE chars
    (the [?] markers count, since they show on screen). first_prefix = extra characters
    shown before the first line (the segment-level "[?] ").
    """
    chunks, cur = [], []
    for w in words:
        prefix = first_prefix if not chunks else 0
        candidate = prefix + len(" ".join(_word_label(x) for x in cur + [w]))
        if cur and (len(cur) >= MAX_WORDS_PER_CUE or candidate > MAX_CHARS_PER_CUE):
            chunks.append(cur)
            cur = []
        cur.append(w)
        # Break early at a comma / full stop once the line has a few words.
        if len(cur) >= SOFT_MIN_WORDS_PER_CUE and w.text[-1:] in ",.?!":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    # Avoid a lonely last word: borrow one from the previous line if it fits.
    if len(chunks) >= 2 and len(chunks[-1]) == 1 and len(chunks[-2]) >= 3:
        moved = [chunks[-2][-1]] + chunks[-1]
        if len(" ".join(_word_label(x) for x in moved)) <= MAX_CHARS_PER_CUE:
            chunks[-2] = chunks[-2][:-1]
            chunks[-1] = moved
    return chunks


def build_cues(segments: list[Segment]) -> list[Cue]:
    cues: list[Cue] = []
    for seg in segments:
        prefix = len(UNSURE) + 1 if seg.low else 0
        for n, chunk in enumerate(chunk_words(seg.hinglish_words, prefix)):
            text = " ".join(_word_label(w) for w in chunk)
            if seg.low and n == 0:
                text = f"{UNSURE} {text}"
            start = int(round(chunk[0].start * 1000))
            end = int(round(chunk[-1].end * 1000))
            cues.append(Cue(start, max(end, start), text))
    return fix_cue_timing(cues)


def fix_cue_timing(cues: list[Cue]) -> list[Cue]:
    """Enforce the minimum duration and remove overlaps (cues may touch, never overlap)."""
    cues.sort(key=lambda c: c.start_ms)
    for i, c in enumerate(cues):
        prev_end = cues[i - 1].end_ms if i else 0
        c.start_ms = max(c.start_ms, prev_end)
        c.end_ms = max(c.end_ms, c.start_ms)
        need = MIN_CUE_DURATION_MS - (c.end_ms - c.start_ms)
        if need > 0:
            # 1) use the gap before the next cue
            next_start = cues[i + 1].start_ms if i + 1 < len(cues) else c.end_ms + need
            grow = min(need, max(0, next_start - c.end_ms))
            c.end_ms += grow
            need -= grow
            # 2) use the gap after the previous cue
            pull = min(need, c.start_ms - prev_end)
            c.start_ms -= pull
            need -= pull
            # 3) otherwise push into the next cue (it gets shifted on the next iteration)
            c.end_ms += max(0, need)
    return cues


def srt_time(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{srt_time(c.start_ms)} --> {srt_time(c.end_ms)}\n{c.text}\n\n"
                   for i, c in enumerate(cues, 1))


def to_transcript(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        line = " ".join(_word_label(w) for w in seg.hinglish_words)
        lines.append(f"{UNSURE} {line}" if seg.low else line)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Transcription (runs in a background thread)
# --------------------------------------------------------------------------------------

_model_cache: dict[tuple[str, str], object] = {}


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_model(size: str, device: str):
    """float16 on an NVIDIA GPU, int8 on CPU. Downloads the model on first use."""
    key = (size, device)
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        compute_type = "float16" if device == "cuda" else "int8"
        _model_cache[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _model_cache[key]


def looks_like_network_error(exc: BaseException) -> bool:
    names = " ".join(t.__name__ for t in type(exc).__mro__)
    text = str(exc).lower()
    return any(n in names for n in ("LocalEntryNotFoundError", "ConnectionError", "ConnectError",
                                    "Timeout", "HfHubHTTPError", "OfflineModeIsEnabled")) \
        or "internet" in text or "connection" in text or "resolve" in text


class UserError(Exception):
    """An error with a friendly title/message for a message box."""

    def __init__(self, title: str, message: str):
        super().__init__(message)
        self.title = title


class TranscriptionJob:
    """Does all the heavy work. Talks to the GUI only through a queue."""

    def __init__(self, audio: Path, model_size: str, api_key: str | None, out: queue.Queue):
        self.audio = audio
        self.model_size = model_size
        self.api_key = api_key           # None = offline mode
        self.q = out

    # -- helpers to talk to the GUI ---------------------------------------------------
    def status(self, text):
        self.q.put(("status", text))

    def progress(self, value):           # None = "busy" animation, else 0..100
        self.q.put(("progress", value))

    # -- main entry point -------------------------------------------------------------
    def run(self):
        try:
            segments = self.transcribe()
            if not segments:
                raise UserError("No speech found",
                                "No speech was detected in this file. Is it the right file?")
            self.convert(segments)
            transcript = to_transcript(segments)
            srt = to_srt(build_cues(segments))
            self.q.put(("done", transcript, srt))
        except UserError as e:
            self.q.put(("error", e.title, str(e)))
        except Exception as e:  # anything unexpected: show it instead of crashing
            traceback.print_exc()
            self.q.put(("error", "Unexpected error", f"{type(e).__name__}: {e}"))

    def transcribe(self) -> list[Segment]:
        devices = ["cuda", "cpu"] if cuda_available() else ["cpu"]
        last_error = None
        for device in devices:
            try:
                return self._transcribe_on(device)
            except UserError:
                raise
            except Exception as e:
                last_error = e
                if device == "cuda":         # e.g. missing cuBLAS/cuDNN DLLs -> retry on CPU
                    self.status("GPU failed, switching to CPU...")
                    continue
        raise last_error

    def _transcribe_on(self, device: str) -> list[Segment]:
        self.progress(None)
        self.status(f"Loading '{self.model_size}' model on {device.upper()} "
                    "(first time: downloading, this can take a while)...")
        try:
            model = load_model(self.model_size, device)
        except Exception as e:
            if device == "cuda":
                raise
            if looks_like_network_error(e):
                raise UserError("Model download failed",
                                f"Couldn't download the '{self.model_size}' model.\n\n"
                                "The first run needs internet to download it (once). "
                                "Check your connection and try again.\n\n"
                                f"Details: {e}") from e
            raise UserError("Model load failed", f"Couldn't load the model.\n\n{e}") from e

        self.status("Reading audio...")
        try:
            seg_iter, info = model.transcribe(
                str(self.audio), language="hi", task="transcribe", beam_size=5,
                vad_filter=True, word_timestamps=True, initial_prompt=WHISPER_PROMPT,
            )
        except Exception as e:
            if device == "cuda":
                raise
            raise UserError("Can't read this file",
                            "This file couldn't be decoded as audio. It may be damaged or "
                            f"in an unsupported format.\n\nDetails: {e}") from e

        duration = max(info.duration, 0.01)
        segments = []
        self.progress(0)
        for s in seg_iter:          # the actual transcription happens while iterating
            words = [Word(w.word.strip(), w.start, w.end, w.probability < LOW_WORD_PROBABILITY)
                     for w in (s.words or []) if w.word.strip()]
            if s.text.strip():
                segments.append(Segment(s.start, s.end, s.text.strip(), words,
                                        low=s.avg_logprob < LOW_SEGMENT_LOGPROB))
            pct = min(100.0, s.end / duration * 100)
            self.progress(pct * 0.9)                    # last 10% = conversion
            self.status(f"Transcribing... {srt_time(int(s.end * 1000))[:8]} / "
                        f"{srt_time(int(duration * 1000))[:8]}")
        return segments

    def convert(self, segments: list[Segment]):
        self.status("Converting to Hinglish...")
        offline = OfflineConverter()

        claude_lines: dict[int, str] = {}
        if self.api_key:
            claude = ClaudeConverter(self.api_key)
            try:
                for b in range(0, len(segments), CLAUDE_BATCH_SIZE):
                    batch = {i: segments[i].text for i in range(b, min(b + CLAUDE_BATCH_SIZE,
                                                                        len(segments)))}
                    self.status(f"Asking Claude for Hinglish... ({b + len(batch)}/{len(segments)})")
                    claude_lines.update(claude.convert(batch))
                    self.progress(90 + 10 * (b + len(batch)) / len(segments))
            except ClaudeError as e:
                self.q.put(("warning", "Claude API problem",
                            f"{e}\n\nThe remaining lines were converted offline instead."))

        for i, seg in enumerate(segments):
            if i in claude_lines:
                seg.hinglish = claude_lines[i]
                seg.hinglish_words = align_by_characters(seg.hinglish, seg.words, seg)
            elif seg.words:
                # Offline: one Devanagari word -> one Hinglish word, so timing is exact.
                seg.hinglish_words = [Word(offline.token(w.text), w.start, w.end, w.low)
                                      for w in seg.words]
                seg.hinglish = " ".join(w.text for w in seg.hinglish_words)
            else:
                seg.hinglish = offline.text(seg.text)
                seg.hinglish_words = proportional_words(seg.hinglish, seg.start, seg.end)
            seg.hinglish_words = merge_punctuation(seg.hinglish_words)
            capitalize_sentences(seg.hinglish_words)
        segments[:] =[s for s in segments if s.hinglish_words]
        self.progress(100)


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.audio: Path | None = None
        self.transcript = ""
        self.srt = ""
        self.q: queue.Queue = queue.Queue()

        root.title(APP_TITLE)
        root.geometry("760x600")
        root.minsize(560, 460)
        root.report_callback_exception = self._on_tk_error

        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        # 1. Upload
        ttk.Button(main, text="Upload Audio", command=self.pick_file).grid(row=0, column=0, sticky="w", **pad)
        self.file_label = ttk.Label(main, text="No file selected", foreground="gray")
        self.file_label.grid(row=0, column=1, columnspan=2, sticky="w", **pad)

        # 2. Model size
        ttk.Label(main, text="Model size:").grid(row=1, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Combobox(main, textvariable=self.model_var, values=MODEL_SIZES, state="readonly",
                     width=12).grid(row=1, column=1, sticky="w", **pad)

        # Claude API option
        self.use_claude = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="Use Claude API for better Hinglish", variable=self.use_claude,
                        command=self._toggle_key).grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        key_row = ttk.Frame(main)
        key_row.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(key_row, text="API key:").pack(side="left")
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(key_row, textvariable=self.key_var, show="•", width=40, state="disabled")
        self.key_entry.pack(side="left", padx=6)
        hint = "(found in ANTHROPIC_API_KEY)" if os.environ.get("ANTHROPIC_API_KEY") else \
            "(or set ANTHROPIC_API_KEY)"
        ttk.Label(key_row, text=hint, foreground="gray").pack(side="left")

        # 3. Transcribe
        self.go_btn = ttk.Button(main, text="Transcribe", command=self.start)
        self.go_btn.grid(row=4, column=0, sticky="w", **pad)

        # 4. Progress + status
        self.bar = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.bar.grid(row=5, column=0, columnspan=3, sticky="ew", **pad)
        self.status_var = tk.StringVar(value="Choose an audio file to start.")
        ttk.Label(main, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky="w", **pad)

        # 5. Read-only transcript box
        box = ttk.Frame(main)
        box.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        main.rowconfigure(7, weight=1)
        self.text = tk.Text(box, wrap="word", height=12, state="disabled", font=("Segoe UI", 10))
        scroll = ttk.Scrollbar(box, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # 6. Output buttons
        btns = ttk.Frame(main)
        btns.grid(row=8, column=0, columnspan=3, sticky="w", **pad)
        self.copy_btn = ttk.Button(btns, text="Copy Text", command=self.copy_text, state="disabled")
        self.txt_btn = ttk.Button(btns, text="Save as .txt", command=lambda: self.save_as(".txt"), state="disabled")
        self.srt_btn = ttk.Button(btns, text="Save as .srt", command=lambda: self.save_as(".srt"), state="disabled")
        for b in (self.copy_btn, self.txt_btn, self.srt_btn):
            b.pack(side="left", padx=(0, 8))

        root.after(300, self.check_ffmpeg)
        root.after(100, self.poll_queue)

    # -- startup check ----------------------------------------------------------------
    def check_ffmpeg(self):
        if shutil.which("ffmpeg") is None:
            messagebox.showwarning(
                "ffmpeg not found",
                "ffmpeg is not installed (or not on PATH).\n\n"
                "Install it by opening a terminal and running:\n\n"
                "    winget install Gyan.FFmpeg\n\n"
                "Then close and reopen this app.")

    # -- UI actions -------------------------------------------------------------------
    def _toggle_key(self):
        self.key_entry.configure(state="normal" if self.use_claude.get() else "disabled")

    def pick_file(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_TYPES)
        if path:
            self.audio = Path(path)
            self.file_label.configure(text=self.audio.name, foreground="")
            self.status_var.set("Ready. Click Transcribe.")

    def start(self):
        if not self.audio:
            messagebox.showinfo(APP_TITLE, "Please upload an audio file first.")
            return
        if not self.audio.is_file() or self.audio.stat().st_size == 0:
            messagebox.showerror("Bad file", f"This file is missing or empty:\n{self.audio}")
            return
        api_key = None
        if self.use_claude.get():
            api_key = self.key_var.get().strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                messagebox.showerror(
                    "API key missing",
                    "Claude API mode needs an Anthropic API key.\n\n"
                    "Paste it into the 'API key' field, or set the ANTHROPIC_API_KEY "
                    "environment variable. Or untick the checkbox to work offline.")
                return

        self._set_busy(True)
        self._show_text("")
        job = TranscriptionJob(self.audio, self.model_var.get(), api_key, self.q)
        threading.Thread(target=job.run, daemon=True).start()

    def copy_text(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.transcript)
        self.status_var.set("Transcript copied to clipboard.")

    def save_as(self, ext: str):
        content = self.transcript if ext == ".txt" else self.srt
        kind = "Text file" if ext == ".txt" else "SubRip subtitles"
        path = filedialog.asksaveasfilename(
            defaultextension=ext, initialdir=str(self.audio.parent),
            initialfile=f"{self.audio.stem}_hinglish{ext}", filetypes=[(kind, f"*{ext}")])
        if path:
            try:
                Path(path).write_text(content, encoding="utf-8")
                self.status_var.set(f"Saved {Path(path).name}")
            except OSError as e:
                messagebox.showerror("Save failed", f"Couldn't save the file:\n{e}")

    # -- worker -> GUI messages -------------------------------------------------------
    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "progress":
                    self._set_progress(msg[1])
                elif kind == "warning":
                    messagebox.showwarning(msg[1], msg[2])
                elif kind == "error":
                    self._set_busy(False)
                    self._set_progress(0)
                    self.status_var.set("Failed. " + msg[1])
                    messagebox.showerror(msg[1], msg[2])
                elif kind == "done":
                    self._finish(msg[1], msg[2])
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def _finish(self, transcript: str, srt: str):
        self.transcript, self.srt = transcript, srt
        self._show_text(transcript)
        self._set_busy(False)
        for b in (self.copy_btn, self.txt_btn, self.srt_btn):
            b.configure(state="normal")
        # Save both files next to the audio file automatically.
        base = self.audio.with_name(f"{self.audio.stem}_hinglish")
        try:
            base.with_suffix(".txt").write_text(transcript, encoding="utf-8")
            base.with_suffix(".srt").write_text(srt, encoding="utf-8")
            self.status_var.set(f"Done. Saved {base.name}.txt and {base.name}.srt next to the audio.")
        except OSError as e:
            self.status_var.set("Done, but auto-save failed. Use the Save buttons.")
            messagebox.showwarning("Auto-save failed", f"Couldn't save next to the audio file:\n{e}")

    # -- small helpers ----------------------------------------------------------------
    def _set_busy(self, busy: bool):
        self.go_btn.configure(state="disabled" if busy else "normal")
        if busy:
            for b in (self.copy_btn, self.txt_btn, self.srt_btn):
                b.configure(state="disabled")

    def _set_progress(self, value):
        if value is None:
            if str(self.bar.cget("mode")) != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(12)
        else:
            if str(self.bar.cget("mode")) != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate")
            self.bar["value"] = value

    def _show_text(self, content: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def _on_tk_error(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        messagebox.showerror("Unexpected error", f"{exc.__name__}: {val}")


def main():
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
