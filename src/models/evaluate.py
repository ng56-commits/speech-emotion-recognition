import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import load_model


# ==========================
# Shared test data (SVM, Random Forest, XGBoost — flat 62-dim features)
# ==========================

X_test = np.load("data/features/X_test.npy")
Y_test = np.load("data/features/Y_test.npy")

results = []


def print_report(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


# ==========================
# 1. SVM (needs scaler)
# ==========================

svm_model = joblib.load("models/svm_model.joblib")
svm_scaler = joblib.load("models/svm_scaler.joblib")

X_test_scaled = svm_scaler.transform(X_test)
svm_predictions = svm_model.predict(X_test_scaled)

print_report("SVM", Y_test, svm_predictions)


# ==========================
# 2. Random Forest (no scaler needed)
# ==========================

rf_model = joblib.load("models/random_forest_model.joblib")
rf_predictions = rf_model.predict(X_test)

print_report("Random Forest", Y_test, rf_predictions)


# ==========================
# 3. XGBoost (needs label encoder — predictions come back as numbers)
# ==========================

xgb_model = joblib.load("models/xgboost_model.joblib")
xgb_label_encoder = joblib.load("models/xgboost_label_encoder.joblib")

xgb_predictions_encoded = xgb_model.predict(X_test)
xgb_predictions = xgb_label_encoder.inverse_transform(xgb_predictions_encoded)

print_report("XGBoost", Y_test, xgb_predictions)


# ==========================
# 4. CNN (different data — spectrograms, not flat features)
# ==========================

X_cnn = np.load("data/features/X_cnn.npy")
y_cnn = np.load("data/features/y_cnn.npy", allow_pickle=True)

X_cnn = X_cnn[..., np.newaxis]

cnn_label_encoder = joblib.load("models/cnn_label_encoder.joblib")
y_cnn_encoded = cnn_label_encoder.transform(y_cnn)

from sklearn.model_selection import train_test_split
_, X_cnn_test, _, y_cnn_test = train_test_split(
    X_cnn, y_cnn_encoded, test_size=0.2, random_state=42, stratify=y_cnn_encoded
)

cnn_model = load_model("models/cnn_model.keras")

cnn_predictions_probs = cnn_model.predict(X_cnn_test)
cnn_predictions_encoded = np.argmax(cnn_predictions_probs, axis=1)

cnn_predictions = cnn_label_encoder.inverse_transform(cnn_predictions_encoded)
cnn_actual = cnn_label_encoder.inverse_transform(y_cnn_test)

print_report("CNN", cnn_actual, cnn_predictions)


# ==========================
# Final Comparison Table
# ==========================

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values(by="f1_score", ascending=False)

print(f"\n{'=' * 50}")
print("FINAL MODEL COMPARISON")
print(f"{'=' * 50}")
print(comparison_df.to_string(index=False))

comparison_df.to_csv("reports/model_comparison.csv", index=False)
print("\nComparison table saved to reports/model_comparison.csv")