from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import logging
import time
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Prediction API")

# ==========================================
# 🛑 THE FIX: DYNAMIC PATH RESOLUTION
# ==========================================
# This ensures it finds the model folder whether running locally or inside Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "titanic_pipeline.pkl")

try:
    model = joblib.load(model_path)
    logger.info("Machine Learning model loaded successfully.")
except Exception as e:
    model = None
    logger.error(f"Failed to load model from {model_path}: {e}")

# Advanced Input Validation Schema
class PassengerData(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Passenger Class (1, 2, or 3)")
    Sex: Literal['male', 'female'] = Field(..., description="Sex ('male' or 'female')")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    Fare: float = Field(..., ge=0, description="Passenger Fare")
    Embarked: Literal['C', 'Q', 'S'] = Field(..., description="Port of Embarkation ('C', 'Q', or 'S')")

# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Time: {process_time:.4f}s")
    return response

# Endpoints
@app.get("/")
def health_check():
    return {"status": "Healthy", "message": "API is running and model path is fixed!"}

@app.post("/predict")
def predict_survival(data: PassengerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        input_df = pd.DataFrame([input_dict])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "prediction": int(prediction),
            "survived": bool(prediction),
            "probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
