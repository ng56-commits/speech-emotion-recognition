from fastapi import APIRouter
from api.schemas import HealthResponse

health_router = APIRouter()

@health_router.get("/health/",response_model = HealthResponse, tags=["health"])
async def health_status():
    return {"status" : "ok"}