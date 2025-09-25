import requests
import json
import sys # For exiting on error if needed

url = "http://127.0.0.1:5000/submit_match" # Change if host/port is different

match_data = {
  "team1_player1": {"name": "Hbox", "character": "Jigglypuff"},
  "team1_player2": {"name": "Plup", "character": "Sheik"},
  "team2_player1": {"name": "Mew2King", "character": "Marth"},
  "team2_player2": {"name": "Leffen", "character": "Fox"},
  "winner": 2
}

headers = {'Content-Type': 'application/json'}

# Define a timeout in seconds.
# You can use a single value for both connect and read timeouts,
# or a tuple (connect_timeout, read_timeout).
request_timeout = 10 # Wait max 10 seconds for the server to respond

print(f"Attempting to POST to {url} with timeout={request_timeout}s...")

try:
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(match_data),
        timeout=request_timeout # Add the timeout parameter here
    )

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    print(f"Success!")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {response.json()}")
    except json.JSONDecodeError:
        print(f"Response Content (not JSON): {response.text}")

except requests.exceptions.Timeout:
    print(f"Error: The request timed out after {request_timeout} seconds.")
    sys.exit(1) # Exit with an error code
except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the server at {url}.")
    print(f"Details: {e}")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"Error: HTTP Error occurred: {e.response.status_code} {e.response.reason}")
    # Try to print the response body if available, it might contain error details
    try:
        print(f"Server Response: {e.response.json()}")
    except json.JSONDecodeError:
        print(f"Server Response (raw): {e.response.text}")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    # Catch any other request-related errors
    print(f"Error: An unexpected error occurred during the request: {e}")
    sys.exit(1)