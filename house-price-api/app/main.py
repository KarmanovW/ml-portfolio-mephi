# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/model.joblib"

app = FastAPI(title="House Price Prediction API")

model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)[0]

    return {"predicted_price": float(prediction)}
