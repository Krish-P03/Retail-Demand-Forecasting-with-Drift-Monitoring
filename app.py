from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("lgbm_model.pkl")
features = joblib.load("features.pkl")

app = FastAPI(title="Retail Demand Forecasting API")

@app.get("/")
def home():
    return {"message": "Retail Demand Forecasting API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = df[features]  
    
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
