# from src.data.single_loader import load_audio
# from src.preprocessing.preprocess import preprocess_pipeline

# audio, sr = load_audio("data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
# audio, sr = preprocess_pipeline(audio, sr)
# print(len(audio), sr)

from src.data.single_loader import load_audio
from src.preprocessing.preprocess import preprocess_pipeline
from src.features.extract_features import extract_mfcc, extract_chroma

audio, sr = load_audio("data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
audio, sr = preprocess_pipeline(audio, sr)

mfcc_vector = extract_mfcc(audio, sr)
chroma_vector = extract_chroma(audio, sr)

print("MFCC shape:", mfcc_vector.shape)      # should be (40,)
print("Chroma shape:", chroma_vector.shape)  # should be (12,)