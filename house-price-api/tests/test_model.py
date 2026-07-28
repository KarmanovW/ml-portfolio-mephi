import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"


def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction_shape():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "MedInc": 8.3,
        "HouseAge": 21,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 900,
        "AveOccup": 2.8,
        "Latitude": 34.2,
        "Longitude": -118.4
    }])

    pred = model.predict(sample)

    assert len(pred) == 1
    assert isinstance(pred[0], float) or isinstance(pred[0], int)
