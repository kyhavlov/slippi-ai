import json
import logging
from collections.abc import Sequence

import requests
from melee import GameState, Character

logger = logging.getLogger(__name__)

url = "http://127.0.0.1:5000/submit_match"  # Change if host/port is different

headers = {'Content-Type': 'application/json'}

# Define a timeout in seconds (connect, read).
request_timeout = 10

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

def character_to_name(character: Character) -> str:
    charname = next(
        name for name, value in vars(Character).items()
        if value == character)
    name = str(charname).capitalize()
    if name == "Cptfalcon":
        name = "Falcon"
    return name


def _build_payload(agent_names: Sequence[str], characters: Sequence[Character], winner: int) -> dict:
    if len(agent_names) < 4 or len(characters) < 4:
        raise ValueError('submit_match expects 4 agent names and characters.')

    ports = [0, 1, 2, 3]
    names_by_port = {port + 1: agent_names[port] for port in ports}
    chars_by_port = {port + 1: character_to_name(characters[port]) for port in ports}

    return {
        "team1_player1": {
            "name": names_by_port.get(1, ""),
            "character": chars_by_port.get(1, ""),
        },
        "team1_player2": {
            "name": names_by_port.get(4, ""),
            "character": chars_by_port.get(4, ""),
        },
        "team2_player1": {
            "name": names_by_port.get(2, ""),
            "character": chars_by_port.get(2, ""),
        },
        "team2_player2": {
            "name": names_by_port.get(3, ""),
            "character": chars_by_port.get(3, ""),
        },
        "winner": winner
    }


def _post_results(payload: dict):
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=request_timeout
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("match reporting request timed out after %ss", request_timeout)
    except requests.exceptions.ConnectionError as e:
        logger.error("match reporting connection error to %s: %s", url, e)
    except requests.exceptions.HTTPError as e:
        try:
            details = e.response.json()
        except json.JSONDecodeError:
            details = e.response.text
        logger.error(
            "match reporting HTTP error %s %s: %s",
            e.response.status_code, e.response.reason, details)
    except requests.exceptions.RequestException as e:
        logger.error("match reporting unexpected error: %s", e)


def submit_match(
    gamestate: GameState,
    agent_names: Sequence[str],
    characters: Sequence[Character],
):
    winner = get_winner(gamestate)
    if winner is None:
        logger.warning("match reporting skipped: sudden death / no winner")
        return

    payload = _build_payload(agent_names, characters, winner)
    _post_results(payload)


def submit_match_summary(
    agent_names: Sequence[str],
    characters: Sequence[Character],
    winner: int,
):
    if winner not in (1, 2):
        logger.warning("match reporting skipped: invalid winner %s", winner)
        return

    payload = _build_payload(agent_names, characters, winner)
    _post_results(payload)
