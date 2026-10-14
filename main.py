import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(
    title="Blood Urgency Prediction API",
    version="1.0"
)

# Debugging ke liye check karein ki files folder mein hain ya nahi
print("Current Directory Files:", os.listdir('.'))

# 1. Model aur Encoders load karna safe tareeqe se
try:
    model = joblib.load('logistic_regression_model.pkl')
    print("urgency_model.pkl successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    encoders = joblib.load('feature_encoders.pkl')
    print("feature_encoders.pkl successfully loaded!")
except Exception as e:
    print(f"Error loading encoders: {e}")

class PatientInput(BaseModel):
    age: int
    gender: str
    weight_kg: float
    blood_group: str
    city: str
    health_issue: str
    hemoglobin_g_dl: float
    platelet_count: int
    systolic_bp: int
    heart_rate: int
    spo2_percent: int
    active_bleeding: int
    previous_transfusions: int
    blood_component_needed: str
    units_required: int
    required_within_hours: int

@app.get("/")
def home():
    return {"message": "API is running live!"}

@app.post("/predict")
def predict_urgency(data: PatientInput):
    # Check karein ki encoders load hue ya nahi
    if 'encoders' not in globals() or encoders is None:
        raise HTTPException(status_code=500, detail="Encoders file could not be loaded on startup.")
    
    try:
        def safe_transform(encoder_key, value):
            if encoder_key not in encoders:
                return 0
            le = encoders[encoder_key]
            if value in le.classes_:
                return le.transform([value])[0]
            else:
                return 0

        input_df = pd.DataFrame([{
            'age': data.age,
            'gender': safe_transform('gender', data.gender),
            'weight_kg': data.weight_kg,
            'blood_group': safe_transform('blood_group', data.blood_group),
            'city': safe_transform('city', data.city),
            'health_issue': safe_transform('health_issue', data.health_issue),
            'hemoglobin_g_dl': data.hemoglobin_g_dl,
            'platelet_count': data.platelet_count,
            'systolic_bp': data.systolic_bp,
            'heart_rate': data.heart_rate,
            'spo2_percent': data.spo2_percent,
            'active_bleeding': data.active_bleeding,
            'previous_transfusions': data.previous_transfusions,
            'blood_component_needed': safe_transform('blood_component_needed', data.blood_component_needed),
            'units_required': data.units_required,
            'required_within_hours': data.required_within_hours
        }])

        prediction_encoded = model.predict(input_df)[0]
        
        target_encoder = encoders.get('urgency_level')
        if target_encoder:
            urgency_result = target_encoder.inverse_transform([prediction_encoded])[0]
        else:
            urgency_result = str(prediction_encoded)

        return {
            "status": "success",
            "predicted_urgency_level": urgency_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
