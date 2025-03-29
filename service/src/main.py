from flask import Flask, request, jsonify, Blueprint, render_template
import numpy as np
import pandas as pd
from model import load

app = Flask(__name__, template_folder='../templates')
xapi_bp = Blueprint('xapi', __name__, url_prefix='/xapi')
v1api_bp = Blueprint('api', __name__, url_prefix='/v1')

model = load()
@v1api_bp.route("/liveness")
def liveness():
    return "live"

@v1api_bp.route("/readiness")
def readiness():
    return f'Model trained: {model != None}'

@v1api_bp.route("/predict",methods=['POST'])
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


def sanitize_input(data):
    """
    Sanitizes the input dictionary by replacing empty strings with None.
    """
    sanitized_data = {}
    for key, value in data.items():
        if isinstance(value, str) and value == '':
            sanitized_data[key] = None
        else:
            sanitized_data[key] = value
    return sanitized_data

@xapi_bp.route('/predict', methods=['POST'])
def predict():
    weather_data = [sanitize_input(request.form.to_dict())]
    try:
        X = pd.DataFrame(weather_data)

        print(weather_data)
        # Ensure correct feature columns
        expected_columns = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                            'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
                            'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
        X = X[expected_columns]

        # Make predictions
        y_pred = model.predict(X)
        return render_template('prediction.html', prediction=y_pred)
    except Exception as e:
        return f"Error calling prediction API: {e}", 500

@app.route('/')
def index():
    return render_template('index.html')

app.register_blueprint(xapi_bp)
app.register_blueprint(v1api_bp)

