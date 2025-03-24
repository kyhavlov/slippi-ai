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
    python analyze_matches.py --view=agent_pair_comps --agent1=Ralph --agent2=Darkatma
    python analyze_matches.py --view=agent_pair_comps --agent1=all --agent2=Ralph
    python analyze_matches.py --view=agent_pair_comps --agent1=all --agent2=all --min-matches=10
    python analyze_matches.py --view=char_agent_winrates --sort-by=trueskill --min-matches=5
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

def analyze_char_agent_winrates(matches: List[Dict], filter_model: Optional[str] = None, min_matches: int = 1, sort_by: str = "win_rate") -> List[Dict]:
    """
    Analyze character + agent name combination win rates from match data.
    
    Args:
        matches: List of match dictionaries
        filter_model: Optional model to filter results for
        min_matches: Minimum number of matches required for inclusion
        sort_by: Field to sort results by ("win_rate" or "trueskill")
    
    Returns:
        List of dictionaries with character + agent name win rate statistics.
    """
    char_agent_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    # For TrueSkill calculation
    import trueskill
    env = trueskill.TrueSkill()
    char_agent_ratings = {}
    
    # First, collect all character+agent combos that appear in any match
    all_char_agent_combos = set()
    for match in matches:
        # Process all combos regardless of filtering
        for i in range(len(match["winner_chars"])):
            if i < len(match["winner_names"]):  # Ensure index is valid
                char = match["winner_chars"][i]
                agent = match["winner_names"][i]
                all_char_agent_combos.add((char, agent))
        
        for i in range(len(match["loser_chars"])):
            if i < len(match["loser_names"]):  # Ensure index is valid
                char = match["loser_chars"][i]
                agent = match["loser_names"][i]
                all_char_agent_combos.add((char, agent))
    
    # Initialize ratings for all combos
    for combo in all_char_agent_combos:
        char_agent_ratings[combo] = env.create_rating()
    
    # Now collect stats with appropriate filtering
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Process winner combinations
        for i in range(len(match["winner_chars"])):
            if i < len(match["winner_names"]):  # Ensure index is valid
                char = match["winner_chars"][i]
                agent = match["winner_names"][i]
                combo = (char, agent)
                
                if not filter_model or winner == filter_model:
                    char_agent_stats[combo]["wins"] += 1
            
        # Process loser combinations
        for i in range(len(match["loser_chars"])):
            if i < len(match["loser_names"]):  # Ensure index is valid
                char = match["loser_chars"][i]
                agent = match["loser_names"][i]
                combo = (char, agent)
                
                if not filter_model or loser == filter_model:
                    char_agent_stats[combo]["losses"] += 1
    
    # Process matches again to update TrueSkill ratings
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Create lists to hold the ratings for the winners and losers
        winner_ratings = []
        winner_combos = []
        loser_ratings = []
        loser_combos = []
        
        # Get winner character+agent combinations and their ratings
        for i in range(len(match["winner_chars"])):
            if i < len(match["winner_names"]):  # Ensure index is valid
                char = match["winner_chars"][i]
                agent = match["winner_names"][i]
                combo = (char, agent)
                winner_combos.append(combo)
                winner_ratings.append(char_agent_ratings[combo])
        
        # Get loser character+agent combinations and their ratings
        for i in range(len(match["loser_chars"])):
            if i < len(match["loser_names"]):  # Ensure index is valid
                char = match["loser_chars"][i]
                agent = match["loser_names"][i]
                combo = (char, agent)
                loser_combos.append(combo)
                loser_ratings.append(char_agent_ratings[combo])
        
        # Skip ratings update if there are no valid combinations
        if not winner_ratings or not loser_ratings:
            continue
        
        # Update ratings based on the match outcome
        rating_groups = [winner_ratings, loser_ratings]
        ranks = [0, 1]  # Lower rank means better performance
        
        try:
            updated_ratings = env.rate(rating_groups, ranks=ranks)
            
            # Update the ratings dictionary with the new ratings
            for i, combo in enumerate(winner_combos):
                char_agent_ratings[combo] = updated_ratings[0][i]
            
            for i, combo in enumerate(loser_combos):
                char_agent_ratings[combo] = updated_ratings[1][i]
        except Exception as e:
            print(f"Warning: Error updating ratings for match ({str(e)})")
            continue
    
    # Calculate win rates and format results
    results = []
    for combo, stats in char_agent_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            rating = char_agent_ratings[combo]
            trueskill_value = rating.mu - 3 * rating.sigma  # Conservative estimate
            
            results.append({
                "character": combo[0],
                "agent": combo[1],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci,
                "trueskill": trueskill_value,
                "mu": rating.mu,
                "sigma": rating.sigma
            })
    
    # Sort by specified field (descending)
    if sort_by == "trueskill":
        return sorted(results, key=lambda x: x["trueskill"], reverse=True)
    else:  # Default to win_rate
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

def print_char_agent_winrates(results: List[Dict], sort_by: str = "win_rate"):
    """Print character + agent combination win rates in a nicely formatted table."""
    sort_method = "TrueSkill rating" if sort_by == "trueskill" else "win rate"
    print(f"\n=== Character + Agent Combination Win Rates (sorted by {sort_method}) ===")
    print(f"{'Character':<15} {'Agent':<20} {'Wins':<7} {'Losses':<7} {'Total':<7} {'Win Rate':<10} {'95% CI':<10} {'TrueSkill':<10}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['character']:<15} {result['agent']:<20} {result['wins']:<7} {result['losses']:<7} {result['total']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%    {result['trueskill']:.2f}")

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

def calculate_overall_char_comp_winrates(matches: List[Dict], filter_model: Optional[str] = None) -> Dict[Tuple[str, str], float]:
    """
    Calculate the overall win rates for each character composition across all agent pairs.
    
    Args:
        matches: List of match dictionaries
        filter_model: Optional filter for specific model
        
    Returns:
        Dictionary mapping character composition tuples to their overall win rates
    """
    # Track stats for each character composition
    comp_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
        
        # Process all character pairs in the winner team
        winner_chars = match["winner_chars"]
        for i in range(len(winner_chars)):
            for j in range(i+1, len(winner_chars)):
                # We need to ensure consistent ordering for character compositions
                char_comp = tuple(sorted([winner_chars[i], winner_chars[j]]))
                comp_stats[char_comp]["wins"] += 1
        
        # Process all character pairs in the loser team
        loser_chars = match["loser_chars"]
        for i in range(len(loser_chars)):
            for j in range(i+1, len(loser_chars)):
                # We need to ensure consistent ordering for character compositions
                char_comp = tuple(sorted([loser_chars[i], loser_chars[j]]))
                comp_stats[char_comp]["losses"] += 1
    
    # Calculate win rates for each character composition
    overall_winrates = {}
    for comp, stats in comp_stats.items():
        total = stats["wins"] + stats["losses"]
        if total > 0:
            win_rate = calculate_win_percentage(stats["wins"], total)
            overall_winrates[comp] = win_rate
    
    return overall_winrates

def analyze_agent_pair_comps(matches: List[Dict], agent1: str, agent2: str, overall_winrates: Dict[Tuple[str, str], float] = None, 
                             filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze character compositions used by a pair of agents when playing together.
    Differentiates which agent played which character.
    
    Args:
        matches: List of match dictionaries
        agent1: First agent name
        agent2: Second agent name
        overall_winrates: Dictionary of overall win rates for each character composition
        filter_model: Optional filter for specific model
        min_matches: Minimum number of matches required for inclusion in results
        
    Returns:
        List of dictionaries with composition statistics for the agent pair
    """
    comp_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue

        # Check if the agent pair is in winner team
        if agent1 in match["winner_names"] and agent2 in match["winner_names"]:
            # Get indices of the agents in the winner team
            idx1 = match["winner_names"].index(agent1)
            idx2 = match["winner_names"].index(agent2)
            
            # Get the characters they were playing
            char1 = match["winner_chars"][idx1]
            char2 = match["winner_chars"][idx2]
            
            # Create a tuple that preserves which agent played which character
            # Format: (agent1_char, agent2_char)
            team_comp = (char1, char2)
            comp_stats[team_comp]["wins"] += 1
            
        # Check if the agent pair is in loser team
        elif agent1 in match["loser_names"] and agent2 in match["loser_names"]:
            # Get indices of the agents in the loser team
            idx1 = match["loser_names"].index(agent1)
            idx2 = match["loser_names"].index(agent2)
            
            # Get the characters they were playing
            char1 = match["loser_chars"][idx1]
            char2 = match["loser_chars"][idx2]
            
            # Create a tuple that preserves which agent played which character
            # Format: (agent1_char, agent2_char)
            team_comp = (char1, char2)
            comp_stats[team_comp]["losses"] += 1
    
    # Format results
    results = []
    for comp, stats in comp_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= min_matches:
            win_rate = calculate_win_percentage(stats["wins"], total)
            ci = confidence_interval(win_rate, total)
            
            # Calculate difference from overall win rate if available
            diff_from_overall = None
            if overall_winrates is not None:
                # Need to find the overall win rate for this character composition
                # We need to sort the characters to match the key in overall_winrates
                sorted_chars = tuple(sorted([comp[0], comp[1]]))
                if sorted_chars in overall_winrates:
                    overall_wr = overall_winrates[sorted_chars]
                    diff_from_overall = win_rate - overall_wr
            
            results.append({
                "agent1_char": comp[0],
                "agent2_char": comp[1],
                "agent1": agent1,
                "agent2": agent2,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": win_rate,
                "confidence": ci,
                "diff_from_overall": diff_from_overall
            })
    
    # Sort by total games (descending), then by win rate (descending)
    return sorted(results, key=lambda x: (x["total"], x["win_rate"]), reverse=True)

def print_agent_pair_comps(results: List[Dict], agent1: str, agent2: str):
    """Print character compositions used by a pair of agents in a nicely formatted table."""
    print(f"\n=== Character Compositions for {agent1} + {agent2} ===")
    print(f"{'Character Assignment':<40} {'Games':<7} {'Wins':<7} {'Losses':<7} {'Win Rate':<10} {'95% CI':<10} {'vs Avg':<10}")
    print("-" * 95)
    
    for result in results:
        char_assignment = f"{agent1} ({result['agent1_char']}) + {agent2} ({result['agent2_char']})"
        diff_str = ""
        if result["diff_from_overall"] is not None:
            diff = result["diff_from_overall"]
            diff_str = f"{diff:+.1f}%" if diff != 0 else "±0.0%"
        
        print(f"{char_assignment:<40} {result['total']:<7} {result['wins']:<7} {result['losses']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%    {diff_str:<10}")
    
    # Print total count and overall win rate
    total_games = sum(result["total"] for result in results)
    total_wins = sum(result["wins"] for result in results)
    total_losses = sum(result["losses"] for result in results)
    overall_win_rate = calculate_win_percentage(total_wins, total_games)
    overall_ci = confidence_interval(overall_win_rate, total_games)
    
    print("-" * 95)
    print(f"{'Overall':<40} {total_games:<7} {total_wins:<7} {total_losses:<7} "
          f"{overall_win_rate:.1f}%    ±{overall_ci:.1f}%")

def analyze_all_agent_pair_comps(matches: List[Dict], specific_agent: Optional[str] = None, 
                                filter_model: Optional[str] = None, min_matches: int = 1) -> List[Dict]:
    """
    Analyze character compositions used by all agent pairs when playing together.
    
    Args:
        matches: List of match dictionaries
        specific_agent: If specified, only analyze pairs that include this agent
        filter_model: Optional filter for specific model
        min_matches: Minimum number of matches required for inclusion in results
        
    Returns:
        List of dictionaries with composition statistics for all agent pairs
    """
    # Calculate overall character composition win rates for comparison
    overall_winrates = calculate_overall_char_comp_winrates(matches, filter_model)
    
    # First, identify all agent pairs that appear in the data
    agent_pairs = set()
    
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
            
        # Get all pairs of agents in the winner team
        for i in range(len(match["winner_names"])):
            for j in range(i+1, len(match["winner_names"])):
                agent1 = match["winner_names"][i]
                agent2 = match["winner_names"][j]
                
                # If specific_agent is provided, only include pairs with that agent
                if specific_agent is None or specific_agent in (agent1, agent2):
                    # Store pairs in a consistent order
                    agent_pairs.add(tuple(sorted([agent1, agent2])))
        
        # Get all pairs of agents in the loser team
        for i in range(len(match["loser_names"])):
            for j in range(i+1, len(match["loser_names"])):
                agent1 = match["loser_names"][i]
                agent2 = match["loser_names"][j]
                
                # If specific_agent is provided, only include pairs with that agent
                if specific_agent is None or specific_agent in (agent1, agent2):
                    # Store pairs in a consistent order
                    agent_pairs.add(tuple(sorted([agent1, agent2])))
    
    # Now analyze each agent pair
    all_results = []
    
    for agent_pair in sorted(agent_pairs):
        agent1, agent2 = agent_pair
        results = analyze_agent_pair_comps(matches, agent1, agent2, overall_winrates, filter_model, min_matches)
        
        if results:  # Only include pairs that have results meeting min_matches criteria
            # Add agent pair info to each result
            for result in results:
                result["agent_pair"] = f"{agent1} + {agent2}"
            
            all_results.extend(results)
    
    # Sort by agent pair, then by total games (descending)
    return sorted(all_results, key=lambda x: (x["agent_pair"], -x["diff_from_overall"]))

def print_all_agent_pair_comps(results: List[Dict]):
    """Print character compositions used by all agent pairs in a nicely formatted table."""
    print("\n=== Character Compositions for All Agent Pairs ===")
    print(f"{'Agent Pair + Characters':<60} {'Games':<7} {'Wins':<7} {'Losses':<7} {'Win Rate':<10} {'95% CI':<10} {'vs Avg':<10}")
    print("-" * 120)
    
    current_pair = None
    pair_totals = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0})
    
    for result in results:
        agent_pair = result["agent_pair"]
        
        # Track totals for each agent pair
        pair_totals[agent_pair]["games"] += result["total"]
        pair_totals[agent_pair]["wins"] += result["wins"]
        pair_totals[agent_pair]["losses"] += result["losses"]
        
        # Print agent pair header if it's a new pair
        if agent_pair != current_pair:
            if current_pair is not None:
                # Print subtotal for previous pair
                prev_totals = pair_totals[current_pair]
                prev_win_rate = calculate_win_percentage(prev_totals["wins"], prev_totals["games"])
                prev_ci = confidence_interval(prev_win_rate, prev_totals["games"])
                print(f"{'SUBTOTAL':<60} {prev_totals['games']:<7} {prev_totals['wins']:<7} "
                      f"{prev_totals['losses']:<7} {prev_win_rate:.1f}%    ±{prev_ci:.1f}%")
                print("-" * 120)
            
            current_pair = agent_pair
        
        # Display character assignment
        char_assignment = f"{result['agent1']} ({result['agent1_char']}) + {result['agent2']} ({result['agent2_char']})"
        
        # Format the difference from overall win rate
        diff_str = ""
        if result["diff_from_overall"] is not None:
            diff = result["diff_from_overall"]
            diff_str = f"{diff:+.1f}%" if diff != 0 else "±0.0%"
        
        print(f"{char_assignment:<60} "
              f"{result['total']:<7} {result['wins']:<7} {result['losses']:<7} "
              f"{result['win_rate']:.1f}%    ±{result['confidence']:.1f}%    {diff_str:<10}")
    
    # Print subtotal for last pair
    if current_pair is not None:
        last_totals = pair_totals[current_pair]
        last_win_rate = calculate_win_percentage(last_totals["wins"], last_totals["games"])
        last_ci = confidence_interval(last_win_rate, last_totals["games"])
        print(f"{'SUBTOTAL':<60} {last_totals['games']:<7} {last_totals['wins']:<7} "
              f"{last_totals['losses']:<7} {last_win_rate:.1f}%    ±{last_ci:.1f}%")
    
    # Print grand total
    total_games = sum(pair["games"] for pair in pair_totals.values())
    total_wins = sum(pair["wins"] for pair in pair_totals.values())
    total_losses = sum(pair["losses"] for pair in pair_totals.values())
    overall_win_rate = calculate_win_percentage(total_wins, total_games)
    overall_ci = confidence_interval(overall_win_rate, total_games)
    
    print("=" * 120)
    print(f"{'GRAND TOTAL':<60} {total_games:<7} {total_wins:<7} {total_losses:<7} "
          f"{overall_win_rate:.1f}%    ±{overall_ci:.1f}%")

def get_unique_agents(matches: List[Dict], filter_model: Optional[str] = None) -> List[str]:
    """Get a list of unique agent names in the match data."""
    agents = set()
    for match in matches:
        winner = extract_model_basename(match["winner"])
        loser = extract_model_basename(match["loser"])
        
        # Skip if filtering by model and neither matches
        if filter_model and filter_model not in (winner, loser):
            continue
            
        for agent in match["winner_names"]:
            agents.add(agent)
        for agent in match["loser_names"]:
            agents.add(agent)
    
    return sorted(list(agents))

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
                 "agent_team_winrates", "char_agent_winrates", "trueskill", 
                 "agent_pair_comps", "all"],
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
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all available agent names in the data"
    )
    parser.add_argument(
        "--agent1",
        help="First agent name for agent pair composition analysis, or 'all' for all agents"
    )
    parser.add_argument(
        "--agent2",
        help="Second agent name for agent pair composition analysis, or 'all' for all agents"
    )
    parser.add_argument(
        "--sort-by",
        choices=["win_rate", "trueskill"],
        default="win_rate",
        help="Field to sort character + agent combinations by"
    )
    
    args = parser.parse_args()
    
    # Load match data
    try:
        matches = load_data(args.data_file)
        print(f"Loaded {len(matches)} matches from {args.data_file}")
    except Exception as e:
        print(f"Error loading match data: {e}")
        return
    
    # Apply filter model if provided
    filter_model = args.filter_model
    if filter_model:
        print(f"Filtering results for model: {filter_model}")
    
    # List available models if requested
    if args.list_models:
        models = get_available_models(matches)
        print("\nAvailable models in the data:")
        for model in models:
            print(f"  {model}")
        return
    
    # List available agents if requested
    if args.list_agents:
        agents = get_unique_agents(matches, filter_model)
        print("\nAvailable agent names in the data:")
        for agent in agents:
            print(f"  {agent}")
        return
    
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
        results = analyze_char_agent_winrates(matches, filter_model, args.min_matches, args.sort_by)
        print_char_agent_winrates(results, args.sort_by)
    
    if args.view == "trueskill" or args.view == "all":
        try:
            results = calculate_trueskill(matches, filter_model)
            print_trueskill_ratings(results)
        except ImportError:
            print("\nError: TrueSkill analysis requires the 'trueskill' package.")
            print("Please install it with: pip install trueskill")
    
    if args.view == "agent_pair_comps":
        if not args.agent1 or not args.agent2:
            print("\nError: --agent1 and --agent2 parameters are required for agent_pair_comps view.")
            print("Example: python analyze_matches.py --view=agent_pair_comps --agent1=Ralph --agent2=Darkatma")
            print("Use 'all' as a wildcard to see all agent combinations:")
            print("Example: python analyze_matches.py --view=agent_pair_comps --agent1=all --agent2=Ralph")
            print("Example: python analyze_matches.py --view=agent_pair_comps --agent1=all --agent2=all --min-matches=10")
            return

        # Calculate overall character composition win rates for comparison
        overall_winrates = calculate_overall_char_comp_winrates(matches, filter_model)
        
        # Handle the 'all' wildcard
        if args.agent1.lower() == 'all' and args.agent2.lower() == 'all':
            # Both agents are 'all' - analyze all agent pairs
            results = analyze_all_agent_pair_comps(matches, None, filter_model, args.min_matches)
            if not results:
                print("\nNo agent pairs found that meet the minimum match criteria.")
                return
            print_all_agent_pair_comps(results)
        elif args.agent1.lower() == 'all':
            # Only agent1 is 'all' - analyze all pairs with agent2
            results = analyze_all_agent_pair_comps(matches, args.agent2, filter_model, args.min_matches)
            if not results:
                print(f"\nNo agent pairs with {args.agent2} found that meet the minimum match criteria.")
                return
            print_all_agent_pair_comps(results)
        elif args.agent2.lower() == 'all':
            # Only agent2 is 'all' - analyze all pairs with agent1
            results = analyze_all_agent_pair_comps(matches, args.agent1, filter_model, args.min_matches)
            if not results:
                print(f"\nNo agent pairs with {args.agent1} found that meet the minimum match criteria.")
                return
            print_all_agent_pair_comps(results)
        else:
            # Neither is 'all' - analyze the specific pair
            results = analyze_agent_pair_comps(matches, args.agent1, args.agent2, overall_winrates, filter_model, args.min_matches)
            if not results:
                print(f"\nNo games found where {args.agent1} and {args.agent2} played together" + 
                      (f" in model {filter_model}" if filter_model else ""))
                return
            print_agent_pair_comps(results, args.agent1, args.agent2)

if __name__ == "__main__":
    main() 