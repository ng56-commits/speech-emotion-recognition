import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_all_datasets
from src.preprocessing.preprocessing import preprocess_pipeline
from src.features.feature_extraction import extract_features


# ----------------------------
# Dataset Paths
# ----------------------------

RAVDESS_PATH = "data/raw/ravdess"
TESS_PATH = "data/raw/tess"


# ----------------------------
# Load all audio
# ----------------------------

audio_data, labels, sample_rates = load_all_datasets(
    RAVDESS_PATH,
    TESS_PATH
)

print(f"Loaded {len(audio_data)} audio files.")


# ----------------------------
# Build X and Y
# ----------------------------

X = []
Y = []

for audio, label, sr in zip(audio_data, labels, sample_rates):

    # Preprocess
    processed_audio, sr = preprocess_pipeline(audio, sr)

    # Feature Extraction
    features = extract_features(processed_audio, sr)

    X.append(features)
    Y.append(label)

# Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

print("Feature Matrix Shape:", X.shape)
print("Labels Shape:", Y.shape)


# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("\nTrain-Test Split")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


# ----------------------------
# Save Dataset
# ----------------------------

save_dir = "data/features"

os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
np.save(os.path.join(save_dir, "Y_test.npy"), Y_test)

print("\nDataset saved successfully!")