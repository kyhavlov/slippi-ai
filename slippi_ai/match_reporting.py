import requests
import json
import sys # For exiting on error if needed
from melee import GameState, Character

url = "http://127.0.0.1:5000/submit_match" # Change if host/port is different

headers = {'Content-Type': 'application/json'}

# Define a timeout in seconds.
# You can use a single value for both connect and read timeouts,
# or a tuple (connect_timeout, read_timeout).
request_timeout = 10 # Wait max 10 seconds for the server to respond

# returns true if one team has no stocks remaining
def match_is_over(gamestate: GameState):
    team_stocks = {0: 0, 1: 0}
        
    # Count stocks for each team
    for port, player in gamestate.players.items():
        if port in (1, 4):
            team_stocks[0] += player.stock
        elif port in (2, 3):
            team_stocks[1] += player.stock

    return team_stocks[0] == 0 or team_stocks[1] == 0

def get_winner(gamestate: GameState):
    # Count stocks for each team
    for port, player in gamestate.players.items():
        if port in (1, 4) and player.stock > 0:
            return 1
        elif port in (2, 3) and player.stock > 0:
            return 2
        
    return None # draw/sudden death

def submit_match(gamestate: GameState, agent_names: tuple[str, str], characters: list[Character]):
    winner = get_winner(gamestate)
    if winner is None:
        print("Error: No winner found! sudden death?")
        return
    
    def character_to_name(character: Character):
        charname = next(name for name, value in vars(Character).items() if value == character)
        name = str(charname).capitalize()

        # special case for characters that have a space in their name
        if name == "Cptfalcon":
            name = "Falcon"

        return name
    
    match_results = {
        "team1_player1": {"name": agent_names[0], "character": character_to_name(characters[0])},
        "team1_player2": {"name": agent_names[3], "character": character_to_name(characters[3])},
        "team2_player1": {"name": agent_names[1], "character": character_to_name(characters[1])},
        "team2_player2": {"name": agent_names[2], "character": character_to_name(characters[2])},
        "winner": winner
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(match_results),
            timeout=request_timeout # Add the timeout parameter here
        )

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        '''print(f"Success!")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response JSON: {response.json()}")
        except json.JSONDecodeError:
            print(f"Response Content (not JSON): {response.text}")'''

    except requests.exceptions.Timeout:
        print(f"Error: The request timed out after {request_timeout} seconds.")
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Could not connect to the server at {url}.")
        print(f"Details: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP Error occurred: {e.response.status_code} {e.response.reason}")
        # Try to print the response body if available, it might contain error details
        try:
            print(f"Server Response: {e.response.json()}")
        except json.JSONDecodeError:
            print(f"Server Response (raw): {e.response.text}")
    except requests.exceptions.RequestException as e:
        # Catch any other request-related errors
        print(f"Error: An unexpected error occurred during the request: {e}")
