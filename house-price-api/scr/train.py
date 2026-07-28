# src/train.py

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from config import MODEL_PATH, DATA_PATH, TARGET_COLUMN
from preprocess import build_preprocessor, split_data


def train():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)

    print("[INFO] Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    print("[INFO] Building model...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("==== Model Metrics ====")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R2   = {r2:.4f}")

    print("[INFO] Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"[OK] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
