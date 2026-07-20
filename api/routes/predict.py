from fastapi import APIRouter,UploadFile,File
from api.schemas import PredictionResponse
from src.data.single_loader import load_audio
from src.preprocessing.preprocess import preprocess_pipeline
import shutil
import os

predict_router=APIRouter()

@predict_router.post("/predict/", tags=["predict"],response_model=PredictionResponse)
async def predict_emotion(file : UploadFile= File()):
    save_path = os.path.join("data/uploads", file.filename)
    with open(save_path, "wb") as buffer:
     shutil.copyfileobj(file.file, buffer)

    emotion = "angry"

    return {
    "predicted_emotion": emotion,
    "confidence": {"angry": 0.9, "happy": 0.1}
}