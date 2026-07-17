#import training and testing data 

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os

# 1. Load the training and testing data
X_train = np.load("data/features/X_train.npy")
X_test = np.load("data/features/X_test.npy")
Y_train = np.load("data/features/Y_train.npy")
Y_test = np.load("data/features/Y_test.npy")

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform on train
X_test_scaled = scaler.transform(X_test)   # transform only, using the same scaler

# 3. Train SVM
model = SVC(kernel="rbf", probability=True)   
model.fit(X_train_scaled,Y_train)

# 4. Save model + scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_model.joblib")
joblib.dump(scaler, "models/svm_scaler.joblib")

print("SVM training complete.")