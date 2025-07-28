import os
import torch
import librosa
import soundfile as sf
from transformers import pipeline
from dotenv import load_dotenv

# Load .env file to get Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file.")

# Load ASR pipeline
pipe = pipeline("automatic-speech-recognition",
                model="openai/whisper-small",
                chunk_length_s=30,
                return_timestamps=True,
                token=hf_token)

# Function to transcribe audio
def transcribe_audio(file_path):
    print("Loading audio:", file_path)
    audio, sr = librosa.load(file_path, sr=16000)  # Whisper expects 16kHz audio
    sf.write("converted.wav", audio, sr)  # Convert to proper format just in case
    result = pipe("converted.wav", generate_kwargs={"language": "ne"})
    return result["text"]

# For testing only
if __name__ == "__main__":
    path = "your_audio.wav"  # Replace this with your own audio file
    transcription = transcribe_audio(path)
    print("Transcription:\n", transcription)
