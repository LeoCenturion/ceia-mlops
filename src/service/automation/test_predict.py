import requests
import json

BASE_URL = "http://127.0.0.1:5000"

test_data = [
    {
        "Date": "2008-12-01",
        "Location": "Albury",
        "MinTemp": 13.4,
        "MaxTemp": 22.9,
        "Rainfall": 0.6,
        "Evaporation": None,
        "Sunshine": None,
        "WindGustDir": "W",
        "WindGustSpeed": 44,
        "WindDir9am": "W",
        "WindDir3pm": "WNW",
        "WindSpeed9am": 20,
        "WindSpeed3pm": 24,
        "Humidity9am": 71,
        "Humidity3pm": 22,
        "Pressure9am": 1007.7,
        "Pressure3pm": 1007.1,
        "Cloud9am": 8,
        "Cloud3pm": None,
        "Temp9am": 16.9,
        "Temp3pm": 21.8,
        "RainToday": "No"
    }
]


def test_predict_endpoint_returns_expected_format():
    """
    Checks expected return format
    """
    response = requests.post(f"{BASE_URL}/predict", json=test_data)


    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    response_json = response.json()
    assert "predictions" in response_json, "Response does not contain 'predictions'"
    assert isinstance(response_json["predictions"], list), "'predictions' should be a list"
    assert len(response_json["predictions"]) == len(test_data), "Mismatch in number of predictions"


def test_predicts_correctly():
    """
    Checks correctly predicts observation
    """
    # Define sample input data


    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    response_json = response.json()

    assert response_json["predictions"] == [0], "Mismatch in expected prediction"

