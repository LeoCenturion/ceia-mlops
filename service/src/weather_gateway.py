from pandas.core.arraylike import default_array_ufunc
import requests

def adapt_weather_data(weather_response):
    """
    Adapts the weather response to the specified schema.
    """
    adapted_data = {
        "Date": weather_response.get("LocalObservationDateTime", "").split("T")[0],
        "Location": "Albury",  # Assuming location is always Albury
        "MinTemp": weather_response.get("TemperatureSummary", {}).get("Past24HourRange", {}).get("Minimum", {}).get("Metric", {}).get("Value"),
        "MaxTemp": weather_response.get("TemperatureSummary", {}).get("Past24HourRange", {}).get("Maximum", {}).get("Metric", {}).get("Value"),
        "Rainfall": weather_response.get("PrecipitationSummary", {}).get("Precipitation", {}).get("Metric", {}).get("Value"),
        "Evaporation": None,
        "Sunshine": None,
        "WindGustDir": weather_response.get("WindGust", {}).get("Direction", {}).get("English"),
        "WindGustSpeed": weather_response.get("WindGust", {}).get("Speed", {}).get("Metric", {}).get("Value"),
        "WindDir9am": weather_response.get("Wind", {}).get("Direction", {}).get("English"),
        "WindDir3pm": weather_response.get("Wind", {}).get("Direction", {}).get("English"),
        "WindSpeed9am": weather_response.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value"),
        "WindSpeed3pm": weather_response.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value"),
        "Humidity9am": weather_response.get("RelativeHumidity"),
        "Humidity3pm": weather_response.get("RelativeHumidity"), # No 3pm humidity, using general
        "Pressure9am": weather_response.get("Pressure", {}).get("Metric", {}).get("Value"),
        "Pressure3pm": weather_response.get("Pressure", {}).get("Metric", {}).get("Value"), #No 3pm pressure, using general
        "Cloud9am": weather_response.get("CloudCover"),
        "Cloud3pm": None,
        "Temp9am": weather_response.get("Temperature", {}).get("Metric", {}).get("Value"),
        "Temp3pm": weather_response.get("Temperature", {}).get("Metric", {}).get("Value"), # No 3pm temp, using general
        "RainToday": "Yes" if weather_response.get("PrecipitationSummary", {}).get("Precipitation", {}).get("Metric", {}).get("Value", 0) > 0 else "No"
    }
    return adapted_data



class WeatherGateway():
    def __init__(self, api_key, base_url="http://dataservice.accuweather.com") -> None:
        self.base_url = base_url
        self.api_key = api_key

    def get_location_key(self, lat, lon):
        url = f"{self.base_url}/locations/v1/cities/geoposition/search"
        params = {
            "apikey": self.api_key,
            "q": f"{lat},{lon}"
        }
        key = None
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            key = data['Key']
        else:
            raise Exception(f"Error: {response.status_code}")
        return key

    def get_current_weather_details(self, lat, lon):
        key = self.get_location_key(lat, lon)
        url = f"{self.base_url}/currentconditions/v1/{key}"
        params = {
            "apikey": self.api_key,
            "details": "true"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return adapt_weather_data(data[0])
        else:
            raise Exception(f"Error: {response.status_code}")

