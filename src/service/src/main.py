from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from model import load
app = Flask(__name__)
model = load()

@app.route("/liveness")
def liveness():
    return "live"

@app.route("/readiness")
def readiness():
    return f'Model trained: {model != None}'

@app.route("/predict",methods=['POST'])
def predict():
    try:
        # Parse JSON input
        req_data = request.get_json()
        # Convert to DataFrame
        X = pd.DataFrame(req_data)

        # Ensure correct feature columns
        expected_columns = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                            'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
                            'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
        X = X[expected_columns]

        # Make predictions
        y_pred = model.predict(X)
        return jsonify({"predictions": y_pred.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
