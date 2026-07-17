import os
import sys
from pathlib import Path

try:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install it with: pip install -r requirements.txt"
    ) from exc

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_all_datasets
from src.preprocessing.preprocessing import preprocess_pipeline


# ----------------------------
# Dataset Paths
# ----------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
RAVDESS_PATH = ROOT_DIR / "data" / "raw" / "ravdess"
TESS_PATH = ROOT_DIR / "data" / "raw" / "tess"

OUTPUT_DIR = ROOT_DIR / "data" / "spectrograms"


# ----------------------------
# Load Audio
# ----------------------------

audio_data, labels, sample_rates = load_all_datasets(
    RAVDESS_PATH,
    TESS_PATH
)

print(f"Loaded {len(audio_data)} files.")


# ----------------------------
# Create emotion folders
# ----------------------------

emotions = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

for emotion in emotions:
    (OUTPUT_DIR / emotion).mkdir(parents=True, exist_ok=True)


# ----------------------------
# Generate Spectrograms
# ----------------------------

emotion_counter = {}

for audio, label, sr in zip(audio_data, labels, sample_rates):

    # preprocess exactly like ML models
    audio, sr = preprocess_pipeline(audio, sr)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )
    
    # unique filename
    count = emotion_counter.get(label, 0)
    emotion_counter[label] = count + 1

    save_path = OUTPUT_DIR / label / f"{label}_{count}.png"

    plt.figure(figsize=(3,3))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis=None,
        y_axis=None
    )

    plt.axis("off")

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

print("Finished generating spectrograms!")