from flask import Flask, request, jsonify, Blueprint, render_template
import pandas as pd
from model import load, read_coords
from weather_gateway import WeatherGateway

# is this a hardcoded api key on a public repository? yes, yes it is. Don't think about it.
ACCUWEATHER_KEY="A8G9Cp9ipeF5CqW4zQcxEqA73fNuYkt0"

app = Flask(__name__, template_folder='../templates')
xapi_bp = Blueprint('xapi', __name__, url_prefix='/xapi')
v1api_bp = Blueprint('api', __name__, url_prefix='/v1')
wg = WeatherGateway(ACCUWEATHER_KEY)
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
        req_data = request.get_json()
        X = pd.DataFrame(req_data)

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
    city_name = request.form.to_dict()['Location']
    coords = read_coords().set_index('city_ascii')
    city = coords.loc[city_name]
    weather_data = wg.get_current_weather_details(city['lat'], city['lng'])

    try:
        X = pd.DataFrame([weather_data])
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

