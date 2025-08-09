import os
import tempfile
import streamlit as st
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv
from pydub import AudioSegment
from pyannote.audio import Pipeline
from datetime import timedelta

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Please add it to your .env file.")
    st.stop()

# Load ASR model
@st.cache_resource
def load_whisper_model():
    try:
        processor = WhisperProcessor.from_pretrained(
            "kiranpantha/whisper-medium-np", use_auth_token=HF_TOKEN
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            "kiranpantha/whisper-medium-np", use_auth_token=HF_TOKEN
        )
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"⚠️ Failed to load Whisper model: {e}")
        return None, None

# Load diarization model
@st.cache_resource
def load_diarization_model():
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
        )
        return pipeline
    except Exception as e:
        st.error(f"⚠️ Failed to load diarization model: {e}")
        return None

# Convert MP3 to WAV (mono, 16kHz)
def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# Run diarization + ASR per segment
def transcribe_segments(wav_path, diarization, processor, model):
    waveform, sr = torchaudio.load(wav_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_waveform = waveform[:, start_sample:end_sample]

        # Skip if too short
        if segment_waveform.shape[1] < 1600:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            torchaudio.save(temp_audio.name, segment_waveform, sr)

            inputs = processor(
                segment_waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                language="ne",
                task="transcribe",
            )

            with torch.no_grad():
                prediction = model.generate(inputs["input_features"], max_new_tokens=128)

            text = processor.batch_decode(prediction, skip_special_tokens=True)[0]
            results.append((speaker, turn.start, turn.end, text.strip()))

    return results

# Display transcript as chat
def display_chat(results):
    st.markdown("### 💬 Conversation Transcript")
    for speaker, start, end, text in results:
        if text:
            time_range = f"[{str(timedelta(seconds=int(start)))} - {str(timedelta(seconds=int(end)))}]"
            st.markdown(f"**🗣 {speaker} {time_range}**\n\n{text}  \n")

# Streamlit UI
st.set_page_config(page_title="Nepali Voice Transcript Tool", layout="centered")
st.markdown("# 🎙 Nepali ASR + Diarization")
st.markdown("Upload your Nepali MP3 audio file. We'll identify speakers and transcribe their speech.")

uploaded_file = st.file_uploader("📤 Upload a Nepali MP3 file", type=["mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/mp3")
    wav_path = convert_mp3_to_wav(tmp_path)

    st.info("📦 Loading models...")
    processor, model = load_whisper_model()
    diarization_model = load_diarization_model()

    if not processor or not model or not diarization_model:
        st.error("❌ Failed to load required models. Check your memory and HF_TOKEN.")
        st.stop()

    st.info("🔍 Running speaker diarization...")
    try:
        diarization = diarization_model(wav_path)
    except Exception as e:
        st.error(f"⚠️ Diarization failed: {e}")
        st.stop()

    st.info("📝 Transcribing by speaker segments...")
    try:
        results = transcribe_segments(wav_path, diarization, processor, model)
        st.success("✅ Done!")
        display_chat(results)
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
