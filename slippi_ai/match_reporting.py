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
    name = character.name.title().replace('_', ' ')
    if name == "Cptfalcon":
        name = "Falcon"
    return name


def _build_payload(
    agent_names: Sequence[str],
    characters: Sequence[Character],
    winner: int,
    *,
    is_teams: bool,
) -> dict:
    if len(agent_names) < 4 or len(characters) < 4:
        raise ValueError('submit_match expects 4 agent names and characters.')

    ports = [0, 1, 2, 3]
    names_by_port = {port + 1: agent_names[port] for port in ports}
    chars_by_port = {port + 1: character_to_name(characters[port]) for port in ports}

    def _player_payload(port: int) -> dict:
        return {
            "name": names_by_port.get(port, ""),
            "character": chars_by_port.get(port, ""),
        }

    if is_teams:
        team1_ports = (1, 4)
        team2_ports = (2, 3)
    else:
        team1_ports = (1,)
        team2_ports = (2,)

    def _team_players(port_tuple: Sequence[int]) -> tuple[dict, dict]:
        players = [_player_payload(port) for port in port_tuple]
        if len(players) == 1:
            players.append({"name": "", "character": ""})
        return players[0], players[1]

    team1_player1, team1_player2 = _team_players(team1_ports)
    team2_player1, team2_player2 = _team_players(team2_ports)

    return {
        "team1_player1": team1_player1,
        "team1_player2": team1_player2,
        "team2_player1": team2_player1,
        "team2_player2": team2_player2,
        "winner": winner,
        "mode": "doubles" if is_teams else "singles",
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
    *,
    is_teams: bool = True,
):
    winner = get_winner(gamestate)
    if winner is None:
        logger.warning("match reporting skipped: sudden death / no winner")
        return

    payload = _build_payload(agent_names, characters, winner, is_teams=is_teams)
    _post_results(payload)


def submit_match_summary(
    agent_names: Sequence[str],
    characters: Sequence[Character],
    winner: int,
    *,
    is_teams: bool,
):
    if winner not in (1, 2):
        logger.warning("match reporting skipped: invalid winner %s", winner)
        return

    payload = _build_payload(agent_names, characters, winner, is_teams=is_teams)
    _post_results(payload)
