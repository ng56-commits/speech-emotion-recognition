import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Load the saved spectrogram data
X_cnn = np.load("data/features/X_cnn.npy")
y_cnn = np.load("data/features/y_cnn.npy")

# 2. Add the "channel" dimension CNNs expect
X_cnn = X_cnn[..., np.newaxis]   

# 3. Encode string labels ("angry", "happy"...) into numbers, then one-hot
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_cnn)
y_onehot = to_categorical(y_encoded)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Build the CNN
num_classes = y_onehot.shape[1]

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 130, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6. Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# 7. Save
model.save("models/cnn_model.keras")
import joblib
joblib.dump(label_encoder, "models/cnn_label_encoder.joblib")

print("CNN training complete.")