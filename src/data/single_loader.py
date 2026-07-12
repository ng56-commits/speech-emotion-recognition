import os
import librosa


def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr