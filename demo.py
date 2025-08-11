import os
import tempfile
import streamlit as st
from datetime import timedelta

# Streamlit UI
st.set_page_config(page_title="Nepali Voice Transcript Tool (Demo)", layout="centered")
st.markdown("# 🎙 Nepali ASR + Diarization")
st.markdown("Upload your Nepali MP3 audio file. We'll show a dmmy conversation between Nepali speakers.")

uploaded_file = st.file_uploader("📤 Upload a Nepali MP3 file", type=["mp3"])

def display_dummy_chat():
    dummy_results = [
        ("Speaker 1", 0, 5, "नमस्ते! तपाईंलाई कस्तो छ?"),
        ("Speaker 2", 5, 10, "म ठिक छु, धन्यवाद। तपाईंलाई कस्तो छ?"),
        ("Speaker 1", 10, 15, "म पनि राम्रो छु। आजको मौसम कस्तो छ?"),
        ("Speaker 2", 15, 20, "आज मौसम धेरै राम्रो छ, घुम्न जान राम्रो दिन छ।"),
    ]
    st.markdown("### 💬 Conversation Transcript")
    for speaker, start, end, text in dummy_results:
        time_range = f"[{str(timedelta(seconds=int(start)))} - {str(timedelta(seconds=int(end)))}]"
        st.markdown(f"**🗣 {speaker} {time_range}**\n\n{text}  \n")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/mp3")
    st.info("🔊 Audio uploaded successfully!")
    
    # Instead of processing, just show dummy transcript
    display_dummy_chat()
else:
    st.info("Please upload a Nepali MP3 file to see the dummy transcript.")
