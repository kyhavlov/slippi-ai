#!/usr/bin/env python
"""
Analyze match records from model pool evaluation.

This script analyzes match data from the model pool evaluation system and
calculates various statistics including character win rates, team composition
win rates, agent name win rates, and agent name team composition win rates.

Example usage:
    python analyze_matches.py --data-file=../eval/model_pool_results/match_records.json --view=character_winrates
    python analyze_matches.py --view=agent_winrates --filter-model=rl_doubles_v4_355.pkl
    python analyze_matches.py --view=team_comp_winrates --min-matches=5
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Set, Optional, Counter, Any
from collections import defaultdict, Counter
import math
import re
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Default path to match records relative to project root
DEFAULT_DATA_FILE = os.path.join(PROJECT_ROOT, "eval", "model_pool_results", "match_records.json")

def load_data(file_path: str) -> List[Dict]:
    """Load match records from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_model_basename(model_path: str) -> str:
    """Extract the model basename from a full path."""
    return os.path.basename(model_path)

def calculate_win_percentage(wins: int, total: int) -> float:
    """Calculate win percentage."""
    if total == 0:
        return 0.0
    return (wins / total) * 100.0

def confidence_interval(win_rate: float, n: int) -> float:
    """Calculate 95% confidence interval for a win rate."""
    if n == 0:
        return 0.0
    # Standard error of a proportion
    std_error = math.sqrt((win_rate / 100.0 * (1 - win_rate / 100.0)) / n)
    # 95% confidence interval (1.96 is the z-score for 95% CI)
    return 1.96 * std_error * 100.0

def analyze_character_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze character win rates from match data.
    
    Returns:
        List of dictionaries with character win rate statistics.
    """
    char_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
            
        # If filtering by model, only count wins/losses for that model
        if filter_model:
            if winner == filter_model:
                # Add win for each winner character
                for char in match["winner_chars"]:
                    char_stats[char]["wins"] += 1
                # Add loss for each loser character
                for char in match["loser_chars"]:
                    char_stats[char]["losses"] += 1
            elif loser == filter_model:
                # Add loss for each loser character
                for char in match["loser_chars"]:
                    char_stats[char]["losses"] += 1
                # Add win for each winner character
                for char in match["winner_chars"]:
                    char_stats[char]["wins"] += 1
        else:
            # No filter, count all characters
            for char in match["winner_chars"]:
                char_stats[char]["wins"] += 1
            for char in match["loser_chars"]:
                char_stats[char]["losses"] += 1
    
    # Calculate win rates and format results
    results = []
    for char, stats in char_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            results.append({
                "character": char,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci
            })
    
    # Sort by win rate (descending)
    return sorted(results, key=lambda x: x["win_rate"], reverse=True)

def analyze_team_comp_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze team composition win rates from match data.
    
    Returns:
        List of dictionaries with team composition win rate statistics.
    """
    team_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Create sorted team compositions
        winner_comp = tuple(sorted(match["winner_chars"]))
        loser_comp = tuple(sorted(match["loser_chars"]))
        
        # If filtering by model, only count wins/losses for that model
        if filter_model:
            if winner == filter_model:
                team_stats[winner_comp]["wins"] += 1
                team_stats[loser_comp]["losses"] += 1
            elif loser == filter_model:
                team_stats[loser_comp]["losses"] += 1
                team_stats[winner_comp]["wins"] += 1
        else:
            # No filter, count all team compositions
            team_stats[winner_comp]["wins"] += 1
            team_stats[loser_comp]["losses"] += 1
    
    # Calculate win rates and format results
    results = []
    for comp, stats in team_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            results.append({
                "team_comp": " + ".join(comp),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci
            })
    
    # Sort by win rate (descending)
    return sorted(results, key=lambda x: x["win_rate"], reverse=True)

def analyze_agent_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze agent name win rates from match data.
    
    Returns:
        List of dictionaries with agent name win rate statistics.
    """
    agent_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # If filtering by model, only count wins/losses for that model
        if filter_model:
            if winner == filter_model:
                # Add win for each winner agent
                for agent in match["winner_names"]:
                    agent_stats[agent]["wins"] += 1
                # Add loss for each loser agent
                for agent in match["loser_names"]:
                    agent_stats[agent]["losses"] += 1
            elif loser == filter_model:
                # Add loss for each loser agent
                for agent in match["loser_names"]:
                    agent_stats[agent]["losses"] += 1
                # Add win for each winner agent
                for agent in match["winner_names"]:
                    agent_stats[agent]["wins"] += 1
        else:
            # No filter, count all agents
            for agent in match["winner_names"]:
                agent_stats[agent]["wins"] += 1
            for agent in match["loser_names"]:
                agent_stats[agent]["losses"] += 1
    
    # Calculate win rates and format results
    results = []
    for agent, stats in agent_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            results.append({
                "agent": agent,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci
            })
    
    # Sort by win rate (descending)
    return sorted(results, key=lambda x: x["win_rate"], reverse=True)

def analyze_agent_team_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze agent name team composition win rates from match data.
    
    Returns:
        List of dictionaries with agent team composition win rate statistics.
    """
    team_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Create sorted team compositions
        winner_team = tuple(sorted(match["winner_names"]))
        loser_team = tuple(sorted(match["loser_names"]))
        
        # If filtering by model, only count wins/losses for that model
        if filter_model:
            if winner == filter_model:
                team_stats[winner_team]["wins"] += 1
                team_stats[loser_team]["losses"] += 1
            elif loser == filter_model:
                team_stats[loser_team]["losses"] += 1
                team_stats[winner_team]["wins"] += 1
        else:
            # No filter, count all team compositions
            team_stats[winner_team]["wins"] += 1
            team_stats[loser_team]["losses"] += 1
    
    # Calculate win rates and format results
    results = []
    for team, stats in team_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            results.append({
                "agent_team": " + ".join(team),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci
            })
    
    # Sort by win rate (descending)
    return sorted(results, key=lambda x: x["win_rate"], reverse=True)

def analyze_char_agent_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze character + agent name combination win rates from match data.
    
    Returns:
        List of dictionaries with character + agent name win rate statistics.
    """
    char_agent_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Process winner combinations
        for i in range(len(match["winner_chars"])):
            char = match["winner_chars"][i]
            agent = match["winner_names"][i]
            combo = (char, agent)
            
            if not filter_model or winner == filter_model:
                char_agent_stats[combo]["wins"] += 1
            
        # Process loser combinations
        for i in range(len(match["loser_chars"])):
            char = match["loser_chars"][i]
            agent = match["loser_names"][i]
            combo = (char, agent)
            
            if not filter_model or loser == filter_model:
                char_agent_stats[combo]["losses"] += 1
    
    # Calculate win rates and format results
    results = []
    for combo, stats in char_agent_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            results.append({
                "character": combo[0],
                "agent": combo[1],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci
            })
    
    # Sort by win rate (descending)
    return sorted(results, key=lambda x: x["win_rate"], reverse=True)

def calculate_trueskill(matches: List[Dict], filter_model: Optional[str] = None) -> List[Dict]:
    """
    Calculate TrueSkill ratings for agent names based on match outcomes.
    
    Returns:
        List of dictionaries with agent name TrueSkill ratings.
    """
    import trueskill
    
    # Initialize TrueSkill environment
    env = trueskill.TrueSkill()
    
    # Initialize ratings dictionary
    ratings = {}
    agent_match_counts = Counter()
    
    # Process all matches to calculate TrueSkill ratings
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Create or get ratings for winner agents
        winner_ratings = []
        for agent in match["winner_names"]:
            if agent not in ratings:
                ratings[agent] = env.create_rating()
            winner_ratings.append(ratings[agent])
            agent_match_counts[agent] += 1
        
        # Create or get ratings for loser agents
        loser_ratings = []
        for agent in match["loser_names"]:
            if agent not in ratings:
                ratings[agent] = env.create_rating()
            loser_ratings.append(ratings[agent])
            agent_match_counts[agent] += 1
        
        # Update ratings based on the match outcome - format correctly for the trueskill library
        # trueskill.rate() expects a list of rating groups, where each group is a list/tuple of ratings
        # The first team (index 0) is the winner
        rating_groups = [winner_ratings, loser_ratings]
        ranks = [0, 1]  # Lower rank means better performance (0 = 1st place, 1 = 2nd place)
        
        try:
            updated_ratings = env.rate(rating_groups, ranks=ranks)
            
            # Update the ratings dictionary
            for i, agent in enumerate(match["winner_names"]):
                ratings[agent] = updated_ratings[0][i]
            
            for i, agent in enumerate(match["loser_names"]):
                ratings[agent] = updated_ratings[1][i]
        except Exception as e:
            print(f"Warning: Error updating ratings for match ({str(e)})")
            print(f"Winner ratings: {winner_ratings}, Loser ratings: {loser_ratings}")
            continue
    
    # Convert ratings to a list of dictionaries
    results = []
    for agent, rating in ratings.items():
        # Skip agents with no matches if filtering by model
        if filter_model and agent_match_counts[agent] == 0:
            continue
        
        results.append({
            "agent": agent,
            "mu": rating.mu,
            "sigma": rating.sigma,
            "trueskill": rating.mu - 3 * rating.sigma,  # Conservative estimate
            "matches": agent_match_counts[agent]
        })
    
    # Sort by TrueSkill rating (descending)
    return sorted(results, key=lambda x: x["trueskill"], reverse=True)

def print_character_winrates(results: List[Dict]):
    """Print character win rates in a nicely formatted table."""
    print("\n=== Character Win Rates ===")
    print(f"{'Character':<15} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['character']:<15} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%")

def print_team_comp_winrates(results: List[Dict]):
    """Print team composition win rates in a nicely formatted table."""
    print("\n=== Team Composition Win Rates ===")
    print(f"{'Team Composition':<30} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['team_comp']:<30} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%")

def print_agent_winrates(results: List[Dict]):
    """Print agent name win rates in a nicely formatted table."""
    print("\n=== Agent Name Win Rates ===")
    print(f"{'Agent Name':<20} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['agent']:<20} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%")

def print_agent_team_winrates(results: List[Dict]):
    """Print agent name team composition win rates in a nicely formatted table."""
    print("\n=== Agent Team Composition Win Rates ===")
    print(f"{'Agent Team':<30} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['agent_team']:<30} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%")

def print_char_agent_winrates(results: List[Dict]):
    """Print character + agent combination win rates in a nicely formatted table."""
    print("\n=== Character + Agent Combination Win Rates ===")
    print(f"{'Character':<15} {'Agent':<20} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['character']:<15} {result['agent']:<20} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%")

def print_trueskill_ratings(results: List[Dict]):
    """Print TrueSkill ratings in a nicely formatted table."""
    print("\n=== Agent TrueSkill Ratings ===")
    print(f"{'Agent':<20} {'TrueSkill':<10} {'Mu':<10} {'Sigma':<10} {'Matches':<7}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['agent']:<20} {result['trueskill']:.2f}      {result['mu']:.2f}      {result['sigma']:.2f}      {result['matches']:<7}")

def get_available_models(matches: List[Dict]) -> List[str]:
    """Get a list of unique model basenames in the match data."""
    models = set()
    for match in matches:
        models.add(extract_model_basename(match["winner"]))
        models.add(extract_model_basename(match["loser"]))
    return sorted(list(models))

def main():
    parser = argparse.ArgumentParser(description="Analyze model pool evaluation match records.")
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="Path to match records JSON file"
    )
    parser.add_argument(
        "--view",
        choices=["character_winrates", "team_comp_winrates", "agent_winrates", 
                 "agent_team_winrates", "char_agent_winrates", "trueskill", "all"],
        default="all",
        help="Type of analysis to display"
    )
    parser.add_argument(
        "--filter-model",
        help="Filter results for a specific model (filename only, not full path)"
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=1,
        help="Minimum number of matches required for inclusion in results"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models in the data"
    )
    
    args = parser.parse_args()
    
    # Load match data
    try:
        matches = load_data(args.data_file)
        print(f"Loaded {len(matches)} matches from {args.data_file}")
    except Exception as e:
        print(f"Error loading match data: {e}")
        return
    
    # List available models if requested
    if args.list_models:
        models = get_available_models(matches)
        print("\nAvailable models in the data:")
        for model in models:
            print(f"  {model}")
        return
    
    # Apply filter model if provided
    filter_model = args.filter_model
    if filter_model:
        print(f"Filtering results for model: {filter_model}")
    
    # Run the requested analysis
    if args.view == "character_winrates" or args.view == "all":
        results = analyze_character_winrates(matches, filter_model, args.min_matches)
        print_character_winrates(results)
    
    if args.view == "team_comp_winrates" or args.view == "all":
        results = analyze_team_comp_winrates(matches, filter_model, args.min_matches)
        print_team_comp_winrates(results)
    
    if args.view == "agent_winrates" or args.view == "all":
        results = analyze_agent_winrates(matches, filter_model, args.min_matches)
        print_agent_winrates(results)
    
    if args.view == "agent_team_winrates" or args.view == "all":
        results = analyze_agent_team_winrates(matches, filter_model, args.min_matches)
        print_agent_team_winrates(results)
    
    if args.view == "char_agent_winrates" or args.view == "all":
        results = analyze_char_agent_winrates(matches, filter_model, args.min_matches)
        print_char_agent_winrates(results)
    
    if args.view == "trueskill" or args.view == "all":
        try:
            results = calculate_trueskill(matches, filter_model)
            print_trueskill_ratings(results)
        except ImportError:
            print("\nError: TrueSkill analysis requires the 'trueskill' package.")
            print("Please install it with: pip install trueskill")

if __name__ == "__main__":
    main() 