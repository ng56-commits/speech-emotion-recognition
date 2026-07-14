import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================
# Load Dataset
# ==========================

X_train = np.load("data/features/X_train.npy")
X_test = np.load("data/features/X_test.npy")

y_train = np.load("data/features/y_train.npy", allow_pickle=True)
y_test = np.load("data/features/y_test.npy", allow_pickle=True)

# ==========================
# Encode Labels
# ==========================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Save the encoder for future predictions
joblib.dump(label_encoder, "models/label_encoder.joblib")

print("Emotion Mapping:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} -> {emotion}")

# ==========================
# Create and Train XGBoost Model
# ==========================

model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# ==========================
# Make Predictions
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Evaluate Model
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("XGBoost Confusion Matrix")
plt.show()

# ==========================
# Show Sample Predictions
# ==========================

predicted_emotions = label_encoder.inverse_transform(y_pred)
actual_emotions = label_encoder.inverse_transform(y_test)

print("\nSample Predictions:\n")

for i in range(10):
    print(
        f"Actual: {actual_emotions[i]:<12} "
        f"Predicted: {predicted_emotions[i]}"
    )

# ==========================
# Save Model
# ==========================

joblib.dump(model, "models/xgboost_model.joblib")

print("\nModel saved as models/xgboost_model.joblib")
print("Label encoder saved as models/label_encoder.joblib")