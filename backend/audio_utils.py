from pydub import AudioSegment
import os

def convert_to_wav_mono(input_path):
    output_path = input_path.replace(".mp3", ".wav").replace(".m4a", ".wav")
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(output_path, format="wav")
    return output_path
