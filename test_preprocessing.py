from src.data.single_loader import load_audio
from src.preprocessing.preprocess import preprocess_pipeline

audio, sr = load_audio("data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
audio, sr = preprocess_pipeline(audio, sr)
print(len(audio), sr)