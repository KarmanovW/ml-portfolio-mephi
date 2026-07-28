# src/predict.py

import joblib
import pandas as pd
import argparse

from config import MODEL_PATH


def predict(features: dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MedInc", type=float, required=True)
    parser.add_argument("--HouseAge", type=float, required=True)
    parser.add_argument("--AveRooms", type=float, required=True)
    parser.add_argument("--AveBedrms", type=float, required=True)
    parser.add_argument("--Population", type=float, required=True)
    parser.add_argument("--AveOccup", type=float, required=True)
    parser.add_argument("--Latitude", type=float, required=True)
    parser.add_argument("--Longitude", type=float, required=True)

    args = parser.parse_args()

    features = vars(args)
    result = predict(features)

    print(f"Predicted house price: {result:.4f}")
