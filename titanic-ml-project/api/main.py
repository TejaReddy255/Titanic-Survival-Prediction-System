from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import logging
import time

# ==========================================
# 1. Setup Logging
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Titanic Survival Prediction API (Bonus Version)")

# ==========================================
# 2. Load Model
# ==========================================
model_path = "titanic-ml-project/model/titanic_pipeline.pkl"
try:
    model = joblib.load(model_path)
    logger.info("Machine Learning model loaded successfully.")
except Exception as e:
    model = None
    logger.error(f"Failed to load model: {e}")

# ==========================================
# 3. Advanced Input Validation Schema
# ==========================================
class PassengerData(BaseModel):
    # Literal enforces exact matches, ge/le enforces min/max numerical bounds
    Pclass: Literal[1, 2, 3] = Field(..., description="Passenger Class (1, 2, or 3)")
    Sex: Literal['male', 'female'] = Field(..., description="Sex ('male' or 'female')")
    Age: float = Field(..., ge=0, le=120, description="Age in years (must be between 0 and 120)")
    Fare: float = Field(..., ge=0, description="Passenger Fare (cannot be negative)")
    Embarked: Literal['C', 'Q', 'S'] = Field(..., description="Port of Embarkation ('C', 'Q', or 'S')")

# ==========================================
# 4. Logging Middleware
# ==========================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Time: {process_time:.4f}s")
    return response

# ==========================================
# 5. Endpoints
# ==========================================
@app.get("/")
def health_check():
    return {"status": "Healthy", "message": "API with logging and validation is running!"}

@app.post("/predict")
def predict_survival(data: PassengerData):
    if model is None:
        logger.error("Prediction attempted, but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        input_df = pd.DataFrame([input_dict])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        logger.info(f"Successful prediction for input data. Survived: {bool(prediction)}")
        
        return {
            "prediction": int(prediction),
            "survived": bool(prediction),
            "probability": round(float(probability), 4)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
