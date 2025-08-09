# Nepali ASR + Diarization Demo

This web app performs Nepali speech-to-text and speaker diarization on uploaded MP3 files.

## Features

- Upload MP3 audio
- Transcribe Nepali speech using OpenAI Whisper
- Perform speaker diarization using pyannote.audio

## Setup Instructions

### 1. Clone and Create Environment

```bash
git clone https://github.com/Freezingle/nepali-asr-diarization
cd Nepali_ASR_Demo
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install ffmpeg (if not already)

```bash
sudo apt install ffmpeg   # Ubuntu/Debian
# or download from https://ffmpeg.org/download.html
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Note:

- For diarization, you need a Hugging Face token.
  Create one from: https://huggingface.co/settings/tokens
  Then replace `"YOUR_HF_TOKEN"` in `app.py` with your token.
