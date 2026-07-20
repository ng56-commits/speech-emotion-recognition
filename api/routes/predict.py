from fastapi import APIRouter,UploadFile,File
from api.schemas import PredictionResponse
from src.data.single_loader import load_audio
from src.preprocessing.preprocessing import preprocess_pipeline
from src.features.feature_extraction import extract_features
from fastapi import HTTPException
import joblib
import shutil
import os

predict_router=APIRouter()

model = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/svm_scaler.joblib")

@predict_router.post("/predict/", tags=["predict"],response_model=PredictionResponse)
async def predict_emotion(file : UploadFile= File()):
    if not file.filename.endswith(".wav"):
     raise HTTPException(status_code=400, detail="Only .wav files are supported")
    save_path = os.path.join("data/uploads", file.filename)
    with open(save_path, "wb") as buffer:
     shutil.copyfileobj(file.file, buffer)

    audio, sr = load_audio(save_path)
    audio, sr = preprocess_pipeline(audio, sr)


    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    probabilities = model.predict_proba(features_scaled)[0]

    confidence_dict = dict(zip(model.classes_, probabilities))
    predicted_emotion = max(confidence_dict, key=confidence_dict.get)

    return {
    "predicted_emotion": predicted_emotion,
    "confidence": confidence_dict
}