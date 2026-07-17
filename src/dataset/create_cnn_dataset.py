import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import librosa
import os
from src.data.loader import load_all_datasets
from src.preprocessing.preprocess import preprocess_pipeline


def audio_to_melspectrogram(audio, sr, n_mels=128, max_len=130):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_spec_db = mel_spec_db[:, :max_len]

    return mel_spec_db


RAVDESS_PATH = "data/raw/ravdess"
TESS_PATH = "data/raw/tess"

audio_data, labels, sample_rates = load_all_datasets(RAVDESS_PATH, TESS_PATH)

X_cnn = []
y_cnn = []

for audio, label, sr in zip(audio_data, labels, sample_rates):
    processed_audio, sr = preprocess_pipeline(audio, sr)
    spec = audio_to_melspectrogram(processed_audio, sr)
    X_cnn.append(spec)
    y_cnn.append(label)

X_cnn = np.array(X_cnn)
y_cnn = np.array(y_cnn)

print("X_cnn shape:", X_cnn.shape)

save_dir = "data/features"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "X_cnn.npy"), X_cnn)
np.save(os.path.join(save_dir, "y_cnn.npy"), y_cnn)

print("CNN spectrogram dataset saved successfully!")