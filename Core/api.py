import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class HeartAttackRiskFeatures(BaseModel):
    Age: float
    Sex: int
    Cholesterol: float
    Heart_Rate: float = Field(..., alias="Heart Rate")
    Diabetes: bool
    Family_History: bool = Field(..., alias="Family History")
    Smoking: bool
    Obesity: bool
    Alcohol_Consumption: bool = Field(..., alias="Alcohol Consumption")
    Exercise_Hours_Per_Week: float = Field(..., alias="Exercise Hours Per Week")
    Diet: int
    Previous_Heart_Problems: bool = Field(..., alias="Previous Heart Problems")
    Medication_Use: bool = Field(..., alias="Medication Use")
    Stress_Level: float = Field(..., alias="Stress Level")
    Sedentary_Hours_Per_Day: float = Field(..., alias="Sedentary Hours Per Day")
    BMI: float
    Triglycerides: float
    Physical_Activity_Days_Per_Week: float = Field(..., alias="Physical Activity Days Per Week")
    Sleep_Hours_Per_Day: float = Field(..., alias="Sleep Hours Per Day")
    Systolic_BP: float = Field(..., alias="Systolic BP")
    Diastolic_BP: float = Field(..., alias="Diastolic BP")

app = FastAPI()

@app.post("/predict")
async def predict_risk(features: HeartAttackRiskFeatures):
    try:
        features = features.dict()
        input_features = np.array([list(features.values())]).reshape(1, -1)
        input_data = pd.DataFrame(input_features)
        prediction = bool(model.predict(input_data)[0])
        return {"Heart Attack Risk": prediction}
    except:
        return {"Error": "Something went wrong. Please try again."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
