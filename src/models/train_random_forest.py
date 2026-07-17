import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load the training and testing data
X_train = np.load("data/features/X_train.npy")
X_test = np.load("data/features/X_test.npy")
Y_train = np.load("data/features/Y_train.npy")
Y_test = np.load("data/features/Y_test.npy")

# 2. Train Random forest
model = RandomForestClassifier()   
model.fit(X_train,Y_train)

# 3. Save model 
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.joblib")

print("Random Forest training complete.")