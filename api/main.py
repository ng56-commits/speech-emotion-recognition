from fastapi import FastAPI
from api.routes import health
from api.routes import predict

app = FastAPI()
app.include_router(health.health_router)
app.include_router(predict.predict_router)
