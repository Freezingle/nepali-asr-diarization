from fastapi import FastAPI, UploadFile
from backend.asr import transcribe_nepali
from backend.audio_utils import convert_to_wav_mono
import shutil
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile):
    os.makedirs("input_audio", exist_ok=True)
    file_path = f"input_audio/{file.filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert to mono WAV
    wav_path = convert_to_wav_mono(file_path)

    # Transcribe
    transcript = transcribe_nepali(wav_path)

    return {"transcript": transcript}
