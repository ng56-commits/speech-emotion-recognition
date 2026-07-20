from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: Dict[str, float]


class HealthResponse(BaseModel):
    status: str