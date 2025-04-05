import http.client
import sys

def health_check(host="localhost", port=5000, path="/v1/liveness"):
    """
    Performs a simple HTTP GET request to check the liveness of a service.

    Args:
        host (str): The hostname of the service.
        port (int): The port number of the service.
        path (str): The path to the liveness endpoint.

    Returns:
        int: 0 if the service is healthy (HTTP 200), 1 otherwise.
    """
    try:
        conn = http.client.HTTPConnection(host, port)
        conn.request("GET", path)
        response = conn.getresponse()

        if response.status == 200:
            return 0  # Healthy
        else:
            return 1  # Unhealthy
    except Exception as e:
        print(f"Error during health check: {e}")
        return 1  # Unhealthy

if __name__ == "__main__":
    sys.exit(health_check())
