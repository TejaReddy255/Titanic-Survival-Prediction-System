from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Initialize FastAPI
app = FastAPI(title="Titanic Survival Prediction API", description="API to predict Titanic passenger survival")

# Load the trained model (using the relative path from the Colab root)
model_path = "titanic-ml-project/model/titanic_pipeline.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the input schema using Pydantic for robust validation
class PassengerData(BaseModel):
    Pclass: int = Field(..., description="Passenger Class (1, 2, or 3)")
    Sex: str = Field(..., description="Sex ('male' or 'female')")
    Age: float = Field(..., description="Age in years")
    Fare: float = Field(..., description="Passenger Fare")
    Embarked: str = Field(..., description="Port of Embarkation ('C', 'Q', or 'S')")

# GET / -> Health Check
@app.get("/")
def health_check():
    return {"status": "Healthy", "message": "Titanic ML API is up and running!"}

# POST /predict -> Prediction Module
@app.post("/predict")
def predict_survival(data: PassengerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Convert incoming JSON/dict to a pandas DataFrame for the pipeline
        # Note: Depending on your Pydantic version, you might need data.dict() instead
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Run prediction through the pipeline
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "prediction": int(prediction),
            "survived": bool(prediction),
            "probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
