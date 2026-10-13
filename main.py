import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# FastAPI app initialize karein
app = FastAPI(
    title="Blood Urgency Prediction API",
    description="API to predict blood urgency level based on patient vitals and medical history.",
    version="1.0"
)

# 1. Saved Model aur Encoders load karein
try:
    model = joblib.load('urgency_model.pkl')
    encoders = joblib.load('feature_encoders.pkl')
    print("Model aur Encoders successfully load ho gaye hain!")
except Exception as e:
    print(f"Error loading model files: {e}")

# 2. Incoming Request Data ke liye Pydantic Model (Schema)
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
    return {"message": "Blood Urgency Prediction API is running live!"}

@app.post("/predict")
def predict_urgency(data: PatientInput):
    try:
        # Incoming text data ko LabelEncoder se safely transform karne ka function
        def safe_transform(encoder_key, value):
            le = encoders[encoder_key]
            # Agar koi nayi category aa jaye jo training mein nahi thi, toh handle karne ke liye
            if value in le.classes_:
                return le.transform([value])[0]
            else:
                # Default pehla class ya 0 use kar sakte hain
                return 0

        # Input data ko DataFrame mein convert karein
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

        # Model se prediction nikalwana (Numeric format mein aayega)
        prediction_encoded = model.predict(input_df)[0]
        
        # Numeric result ko wapas original text urgency level (Low, Medium, High, Critical) mein convert karna
        target_encoder = encoders['urgency_level']
        urgency_result = target_encoder.inverse_transform([prediction_encoded])[0]

        return {
            "status": "success",
            "predicted_urgency_level": urgency_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))