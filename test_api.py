import requests
import pytest

API_URL = "http://127.0.0.1:8000/predict"

@pytest.mark.parametrize("text", [
    "Win a free cruise now!",
    "Hi boss, here's the report.",
    "",
    "BUY NOW!!! Special offer, limited time!",
    "<html><body>Hello</body></html>",
    "Ceci est un message",
    "1234567890",
])
def test_api_predictions(text):
    response = requests.post(API_URL, json={"message": text})
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["spam", "not spam"]
    assert 0.0 <= data["confidence"] <= 1.0