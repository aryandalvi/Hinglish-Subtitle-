# Hinglish Transcriber

Desktop app (tkinter) that turns an audio/video file into a Hinglish transcript (`<name>_hinglish.txt`)
and phone-friendly Hinglish subtitles (`<name>_hinglish.srt`), saved next to the audio file.

## Setup (Windows 10/11)

1. Install Python 3.12 (or 3.14) from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Install ffmpeg (once):
   ```
   winget install Gyan.FFmpeg
   ```
   Close and reopen the terminal afterwards.
3. Open **Command Prompt** in this folder and create a virtual environment:
   ```
   cd /d D:\Projects\Transcript-HInglish
   py -3.12 -m venv .venv
   ```
   (Use `py -3.14` for Python 3.14, or plain `python -m venv .venv`.)
4. Activate it and install the packages:
   ```
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the app:
   ```
   python app.py
   ```

Next time, you only need:
```
cd /d D:\Projects\Transcript-HInglish
.venv\Scripts\activate
python app.py
```

## Using it

1. **Upload Audio**: pick an mp3 / wav / m4a / ogg / flac / mp4 file.
2. Choose the model: `small` (fast), `medium` (default), or `large-v3` (best, slow on CPU).
   The first time you use a model it downloads (about 0.5 / 1.5 / 3 GB), so you need internet once.
   After that it works offline.
3. Optionally tick **Use Claude API for better Hinglish**. Paste your key into the field,
   or set it once for your user account:
   ```
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```
   Then open a new terminal.
4. Click **Transcribe**. When it finishes, both files are saved next to the audio. You can also use
   **Copy Text**, **Save as .txt**, or **Save as .srt**.

`[?]` marks words or lines that Whisper wasn't sure about. Check those before you publish.

### Teaching the offline mode new words

Create `hinglish_custom.txt` next to `app.py` (or next to the .exe) with one word per line:
```
# devanagari = how you want it written
इम्प्लांट = implant
मुस्कान = muskaan
```
These entries override the built-in dictionaries.

### GPU (optional)

If you have an NVIDIA GPU, the app uses it automatically (float16). If the CUDA 12 libraries
(cuBLAS and cuDNN 9) aren't installed, it falls back to the CPU (int8) by itself.

## Package as a single .exe (optional)

```
.venv\Scripts\activate
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name HinglishTranscriber ^
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all indic_transliteration ^
  --collect-binaries onnxruntime --collect-binaries av app.py
```
The program is `dist\HinglishTranscriber.exe` (about 110 MB). Whisper models aren't bundled.
They download on first use into `%USERPROFILE%\.cache\huggingface`.
"# Hinglish-Subtitle-" 
