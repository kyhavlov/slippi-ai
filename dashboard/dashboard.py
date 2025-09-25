import os
import json
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from collections import defaultdict
import threading # Basic lock for thread safety on file write / list append
import logging # Import logging

# --- Configuration ---
DATA_DIR = "melee_data"
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f" # ISO 8601 like format

# --- Global State ---
matches_data = [] # Holds all match data in memory
data_lock = threading.Lock() # To protect access to matches_data and file writes

# --- Initialization ---
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Ensure Flask logger uses INFO level

# --- Helper Functions ---

def get_daily_filename(dt):
    """Generates the filename for a given datetime."""
    return os.path.join(DATA_DIR, f"matches_{dt.strftime(DATE_FORMAT)}.jsonl")

def ensure_data_dir():
    """Creates the data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

def save_match(match_record):
    """Appends a match record to the appropriate daily file and in-memory list."""
    ensure_data_dir()
    now = datetime.utcnow()
    match_record["timestamp"] = now.strftime(TIMESTAMP_FORMAT) # Add timestamp before saving

    filename = get_daily_filename(now)
    try:
        with data_lock: # Acquire lock before modifying shared resources
            # Create the parsed datetime object immediately for the in-memory list
            match_record_with_dt = match_record.copy()
            match_record_with_dt["timestamp_dt"] = now

            # Append to file (JSON Lines format)
            with open(filename, 'a') as f:
                # Save the version without the datetime object to JSON
                json.dump(match_record, f)
                f.write('\n')

            # Append to in-memory list (with the datetime object)
            matches_data.append(match_record_with_dt)
            app.logger.info(f"Match recorded. Total matches in memory: {len(matches_data)}") # Log update
        return True
    except IOError as e:
        app.logger.error(f"Error saving match to {filename}: {e}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error saving match: {e}")
        return False

def load_matches():
    """Loads all matches from files in DATA_DIR into memory."""
    ensure_data_dir()
    loaded_matches = []
    app.logger.info(f"Loading match data from {DATA_DIR}...")
    try:
        filenames = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("matches_") and f.endswith(".jsonl")])
        for filename in filenames:
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                match = json.loads(line)
                                # Add parsed timestamp if present
                                if "timestamp" in match:
                                    try:
                                         match["timestamp_dt"] = datetime.strptime(match["timestamp"], TIMESTAMP_FORMAT)
                                         loaded_matches.append(match)
                                    except ValueError:
                                        app.logger.warning(f"Skipping record with invalid timestamp format in {filename} (line {i+1}): {match.get('timestamp')}")
                                else:
                                    app.logger.warning(f"Skipping record without timestamp in {filename} (line {i+1}): {line[:50]}...")
                            except json.JSONDecodeError:
                                app.logger.warning(f"Skipping invalid JSON line in {filename} (line {i+1}): {line[:50]}...")
            except IOError as e:
                app.logger.error(f"Error reading file {filepath}: {e}")
            except Exception as e:
                 app.logger.error(f"Unexpected error processing file {filepath}: {e}")

        # Sort by timestamp after loading everything
        loaded_matches.sort(key=lambda x: x.get("timestamp_dt", datetime.min))

        with data_lock: # Update global list safely
            global matches_data
            matches_data = loaded_matches

        app.logger.info(f"Loaded {len(matches_data)} matches.")

    except Exception as e:
        app.logger.error(f"Failed to list or process files in {DATA_DIR}: {e}")


def calculate_stats(filtered_matches):
    """Calculates character, team comp, player, player+char, and player pairing win rates."""
    # Character stats
    char_wins = defaultdict(int)
    char_games = defaultdict(int)
    # Team composition stats
    team_wins = defaultdict(int)
    team_games = defaultdict(int)
    # Player stats
    player_wins = defaultdict(int)
    player_games = defaultdict(int)
    # Player+Character stats
    player_char_wins = defaultdict(int)
    player_char_games = defaultdict(int)
    # Player pairing stats
    pairing_wins = defaultdict(int)
    pairing_games = defaultdict(int)
    
    # Character-specific stats
    char_teammate_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {char: {teammate: [wins, games]}}
    char_player_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))    # {char: {player: [wins, games]}}
    char_opponent_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {char: {opponent: [wins, games]}}

    for match in filtered_matches:
        try:
            # Extract player and character info
            p11 = match["team1_player1"]
            p12 = match["team1_player2"]
            p21 = match["team2_player1"]
            p22 = match["team2_player2"]

            players_team1 = [p11, p12]
            players_team2 = [p21, p22]

            team1_chars = sorted([p11["character"], p12["character"]])
            team2_chars = sorted([p21["character"], p22["character"]])

            # Canonical team composition (sorted tuple of characters)
            team1_comp = tuple(team1_chars)
            team2_comp = tuple(team2_chars)

            # Canonical player pairing (sorted tuple of names)
            team1_pairing = tuple(sorted([p11["name"], p12["name"]]))
            team2_pairing = tuple(sorted([p21["name"], p22["name"]]))

            winner = match["winner"]
            winning_players = []
            losing_players = []
            winning_chars = []
            losing_chars = []
            winning_comp = None
            losing_comp = None
            winning_pairing = None
            losing_pairing = None

            if winner == 1:
                winning_players = players_team1
                losing_players = players_team2
                winning_chars = team1_chars
                losing_chars = team2_chars
                winning_comp = team1_comp
                losing_comp = team2_comp
                winning_pairing = team1_pairing
                losing_pairing = team2_pairing
            elif winner == 2:
                winning_players = players_team2
                losing_players = players_team1
                winning_chars = team2_chars
                losing_chars = team1_chars
                winning_comp = team2_comp
                losing_comp = team1_comp
                winning_pairing = team2_pairing
                losing_pairing = team1_pairing
            else:
                app.logger.warning(f"Skipping match with invalid winner ({winner}): {match.get('timestamp')}")
                continue # Skip if winner info is invalid

            # --- Update Stats ---

            # Characters
            for char in winning_chars:
                char_wins[char] += 1
                char_games[char] += 1
            for char in losing_chars:
                char_games[char] += 1

            # Character teammate stats (process each character individually)
            for i, char in enumerate(winning_chars):
                teammate = winning_chars[1-i]  # Get the teammate character
                char_teammate_stats[char][teammate][0] += 1  # Win
                char_teammate_stats[char][teammate][1] += 1  # Game
                
            for i, char in enumerate(losing_chars):
                teammate = losing_chars[1-i]  # Get the teammate character
                char_teammate_stats[char][teammate][1] += 1  # Game only, no win
            
            # Character player stats
            for p in winning_players:
                p_name = p["name"]
                p_char = p["character"]
                char_player_stats[p_char][p_name][0] += 1  # Win
                char_player_stats[p_char][p_name][1] += 1  # Game
            
            for p in losing_players:
                p_name = p["name"]
                p_char = p["character"]
                char_player_stats[p_char][p_name][1] += 1  # Game only, no win
            
            # Character opponent stats
            for char in winning_chars:
                for opp_char in losing_chars:
                    char_opponent_stats[char][opp_char][0] += 1  # Win
                    char_opponent_stats[char][opp_char][1] += 1  # Game
            
            for char in losing_chars:
                for opp_char in winning_chars:
                    char_opponent_stats[char][opp_char][1] += 1  # Game only, no win

            # Team Compositions
            if winning_comp:
                team_wins[winning_comp] += 1
                team_games[winning_comp] += 1
            if losing_comp:
                team_games[losing_comp] += 1

            # Players and Player+Character
            for p in winning_players:
                p_name = p["name"]
                p_char = p["character"]
                p_key = f"{p_name} ({p_char})"
                player_wins[p_name] += 1
                player_games[p_name] += 1
                player_char_wins[p_key] += 1
                player_char_games[p_key] += 1
            for p in losing_players:
                p_name = p["name"]
                p_char = p["character"]
                p_key = f"{p_name} ({p_char})"
                player_games[p_name] += 1
                player_char_games[p_key] += 1

            # Player Pairings
            if winning_pairing:
                pairing_wins[winning_pairing] += 1
                pairing_games[winning_pairing] += 1
            if losing_pairing:
                pairing_games[losing_pairing] += 1

        except KeyError as e:
            app.logger.warning(f"Skipping match due to missing key {e}: {match.get('timestamp')}")
            continue
        except Exception as e:
            app.logger.warning(f"Skipping match calculation due to unexpected error {e}: {match.get('timestamp')}")
            continue

    # --- Calculate Rates ---
    def _calculate_rates(wins_dict, games_dict, key_formatter=None):
        rates = {}
        for key, games in games_dict.items():
            wins = wins_dict[key]
            rate = (wins / games * 100) if games > 0 else 0
            display_key = key_formatter(key) if key_formatter else key
            rates[display_key] = {"wins": wins, "games": games, "rate": round(rate, 2)}
        # Sort by win rate descending
        return dict(sorted(rates.items(), key=lambda item: item[1]['rate'], reverse=True))

    char_win_rates = _calculate_rates(char_wins, char_games)
    team_win_rates = _calculate_rates(team_wins, team_games, key_formatter=lambda k: f"{k[0]} / {k[1]}")
    player_win_rates = _calculate_rates(player_wins, player_games)
    player_char_win_rates = _calculate_rates(player_char_wins, player_char_games)
    pairing_win_rates = _calculate_rates(pairing_wins, pairing_games, key_formatter=lambda k: f"{k[0]} / {k[1]}")

    # Process character-specific stats
    char_teammate_win_rates = {}
    char_player_win_rates = {}
    char_opponent_win_rates = {}
    
    for char in char_games:
        # Teammate win rates with difference from overall
        teammate_rates = {}
        # Get character's overall win rate for comparison
        char_overall_rate = char_win_rates.get(char, {}).get('rate', 0)
        
        for teammate, (wins, games) in char_teammate_stats[char].items():
            if games > 0:
                rate = (wins / games) * 100
                # Compare with the character's overall win rate (not the teammate's)
                diff = rate - char_overall_rate
                teammate_rates[teammate] = {
                    "wins": wins, 
                    "games": games, 
                    "rate": round(rate, 2),
                    "diff": round(diff, 2)
                }
        char_teammate_win_rates[char] = dict(sorted(teammate_rates.items(), 
                                                key=lambda item: item[1]['rate'], 
                                                reverse=True))
        
        # Player win rates with difference from overall (this is already correct)
        player_rates = {}
        for player, (wins, games) in char_player_stats[char].items():
            if games > 0:
                rate = (wins / games) * 100
                diff = rate - char_overall_rate
                player_rates[player] = {
                    "wins": wins, 
                    "games": games, 
                    "rate": round(rate, 2),
                    "diff": round(diff, 2)
                }
        char_player_win_rates[char] = dict(sorted(player_rates.items(), 
                                               key=lambda item: item[1]['rate'], 
                                               reverse=True))
        
        # Opponent character win rates with difference from overall (this is already correct)
        opponent_rates = {}
        for opponent, (wins, games) in char_opponent_stats[char].items():
            if games > 0:
                rate = (wins / games) * 100
                diff = rate - char_overall_rate
                opponent_rates[opponent] = {
                    "wins": wins, 
                    "games": games, 
                    "rate": round(rate, 2),
                    "diff": round(diff, 2)
                }
        char_opponent_win_rates[char] = dict(sorted(opponent_rates.items(), 
                                                key=lambda item: item[1]['rate'], 
                                                reverse=True))

    return (
        char_win_rates,
        team_win_rates,
        player_win_rates,
        player_char_win_rates,
        pairing_win_rates,
        char_teammate_win_rates,
        char_player_win_rates,
        char_opponent_win_rates
    )


# --- Routes ---

@app.route('/')
def dashboard():
    """Displays the statistics dashboard."""
    period = request.args.get('period', 'all') # e.g., '1d', '7d', '30d', 'all'
    min_games = request.args.get('min_games', '0')
    
    # Validate and convert min_games to integer
    try:
        min_games = int(min_games)
        if min_games < 0:
            min_games = 0
    except ValueError:
        min_games = 0
    
    now = datetime.utcnow()
    cutoff_dt = datetime.min # Default to beginning of time

    # Determine cutoff time based on period
    if period == '1d': cutoff_dt = now - timedelta(days=1)
    elif period == '2d': cutoff_dt = now - timedelta(days=2)
    elif period == '7d': cutoff_dt = now - timedelta(days=7)
    elif period == '30d': cutoff_dt = now - timedelta(days=30)
    # 'all' uses the default cutoff_dt (datetime.min)

    filtered_matches = []
    total_match_count = 0
    # Filter matches based on the period USING THE LOCK
    with data_lock: # Access matches_data safely
        total_match_count = len(matches_data) # Get current total count safely
        app.logger.info(f"Dashboard request: Period='{period}', Min games='{min_games}'. Processing {total_match_count} total matches.")
        filtered_matches = [
            match for match in matches_data
            if "timestamp_dt" in match and match["timestamp_dt"] >= cutoff_dt
        ]
        app.logger.info(f"Found {len(filtered_matches)} matches within the period.")

    # Calculate all statistics
    (char_stats,
     team_stats,
     player_stats,
     player_char_stats,
     pairing_stats,
     char_teammate_stats,
     char_player_stats,
     char_opponent_stats) = calculate_stats(filtered_matches)
    
    # Apply minimum games filter
    if min_games > 0:
        # Filter character stats
        char_stats = {char: stats for char, stats in char_stats.items() 
                     if stats['games'] >= min_games}
        
        # Filter team stats
        team_stats = {team: stats for team, stats in team_stats.items() 
                     if stats['games'] >= min_games}
        
        # Filter player stats
        player_stats = {player: stats for player, stats in player_stats.items() 
                       if stats['games'] >= min_games}
        
        # Filter player+char stats
        player_char_stats = {p_char: stats for p_char, stats in player_char_stats.items() 
                            if stats['games'] >= min_games}
        
        # Filter pairing stats
        pairing_stats = {pair: stats for pair, stats in pairing_stats.items() 
                        if stats['games'] >= min_games}
        
        # Filter character teammate stats
        for char in list(char_teammate_stats.keys()):
            # Skip characters with too few games
            if char not in char_stats:
                del char_teammate_stats[char]
                continue
                
            # Filter teammates within this character
            char_teammate_stats[char] = {
                teammate: stats for teammate, stats in char_teammate_stats[char].items()
                if stats['games'] >= min_games
            }
        
        # Filter character player stats
        for char in list(char_player_stats.keys()):
            # Skip characters with too few games
            if char not in char_stats:
                del char_player_stats[char]
                continue
                
            # Filter players within this character
            char_player_stats[char] = {
                player: stats for player, stats in char_player_stats[char].items()
                if stats['games'] >= min_games
            }
        
        # Filter character opponent stats
        for char in list(char_opponent_stats.keys()):
            # Skip characters with too few games
            if char not in char_stats:
                del char_opponent_stats[char]
                continue
                
            # Filter opponents within this character
            char_opponent_stats[char] = {
                opponent: stats for opponent, stats in char_opponent_stats[char].items()
                if stats['games'] >= min_games
            }

    return render_template(
        'index.html',
        # Stats data
        char_stats=char_stats,
        team_stats=team_stats,
        player_stats=player_stats,
        player_char_stats=player_char_stats,
        pairing_stats=pairing_stats,
        char_teammate_stats=char_teammate_stats,
        char_player_stats=char_player_stats,
        char_opponent_stats=char_opponent_stats,
        # Page info
        selected_period=period,
        min_games=min_games,
        total_matches=total_match_count, # Use the count obtained under lock
        filtered_matches_count=len(filtered_matches)
    )


@app.route('/submit_match', methods=['POST'])
def submit_match():
    """API endpoint to submit a new match result."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()

    # Basic Validation (can be expanded)
    required_fields = ["team1_player1", "team1_player2", "team2_player1", "team2_player2", "winner"]
    player_fields = ["name", "character"]

    missing_req = [f for f in required_fields if f not in data]
    if missing_req:
         return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_req)}"}), 400

    for team_player_key in ["team1_player1", "team1_player2", "team2_player1", "team2_player2"]:
        player_data = data.get(team_player_key)
        if not isinstance(player_data, dict):
             return jsonify({"status": "error", "message": f"Field '{team_player_key}' must be a JSON object"}), 400
        missing_p_fields = [pf for pf in player_fields if pf not in player_data]
        if missing_p_fields:
             return jsonify({"status": "error", "message": f"Missing fields in '{team_player_key}': {', '.join(missing_p_fields)}"}), 400
        # Ensure values are not empty strings
        if not player_data.get("name", "").strip() or not player_data.get("character", "").strip():
             return jsonify({"status": "error", "message": f"Player name and character cannot be empty in '{team_player_key}'"}), 400

    if data.get("winner") not in [1, 2]:
        return jsonify({"status": "error", "message": "Winner must be 1 or 2"}), 400

    # Prepare the record (perform basic string conversion for safety)
    try:
        match_record = {
            "team1_player1": {"name": str(data["team1_player1"]["name"]).strip(), "character": str(data["team1_player1"]["character"]).strip()},
            "team1_player2": {"name": str(data["team1_player2"]["name"]).strip(), "character": str(data["team1_player2"]["character"]).strip()},
            "team2_player1": {"name": str(data["team2_player1"]["name"]).strip(), "character": str(data["team2_player1"]["character"]).strip()},
            "team2_player2": {"name": str(data["team2_player2"]["name"]).strip(), "character": str(data["team2_player2"]["character"]).strip()},
            "winner": int(data["winner"]),
            # timestamp will be added by save_match
        }
    except Exception as e:
         app.logger.error(f"Error preparing match record from input data: {e}. Data: {data}")
         return jsonify({"status": "error", "message": "Invalid data format during record preparation"}), 400


    if save_match(match_record):
        return jsonify({"status": "success", "message": "Match recorded"}), 201
    else:
        # Log already happened in save_match
        return jsonify({"status": "error", "message": "Internal server error saving match data"}), 500

# --- Run Application ---
if __name__ == '__main__':
    load_matches() # Load existing data on startup
    # Use threaded=True for basic concurrency handling with the lock
    # Use debug=True for development (auto-reloads, provides debugger)
    # Set host='0.0.0.0' to make it accessible on your network
    # Note: Werkzeug reloader (used by debug=True) can sometimes cause issues with global state
    # in more complex scenarios, but should be okay here with the lock.
    # For production, use a proper WSGI server like gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)