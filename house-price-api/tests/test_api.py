from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    payload = {
        "MedInc": 8.3,
        "HouseAge": 21,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 900,
        "AveOccup": 2.8,
        "Latitude": 34.2,
        "Longitude": -118.4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
