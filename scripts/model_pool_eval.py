#!/usr/bin/env python
"""Run a pool of games between AI models for TrueSkill evaluation.

This script continuously runs doubles games between different AI models to calculate
TrueSkill ratings for each model. It uses Ray actors to run games in parallel and
determine winners based on in-game stock counts.

The script tracks model performance over time and maintains TrueSkill ratings
to determine which models perform better than others. Multiple models can be
evaluated simultaneously in a round-robin tournament style.

Features:
- Smart model selection with exploration-exploitation tradeoff
- Prioritizes matchups that provide the most information gain
- Tracks matchup history and ensures balanced comparisons
- Adapts temperature parameter over time to transition from exploration to exploitation
- Character randomization and agent name extraction from model files

Example:
    python scripts/model_pool_eval.py \
        --dolphin_path=/path/to/slippi-dolphin \
        --dolphin_iso=/path/to/SSBM.iso \
        --model_dir=/path/to/models/directory \
        --max_parallel_games=3
"""

import logging
import os
import random
import subprocess
import sys
import time
import tempfile
import glob
from typing import Dict, List, Optional, Tuple
import json
import math

import melee
import ray
from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import utils, eval_lib, saving
from slippi_ai.trueskill_ranker import ModelRanker
from slippi_ai.game_runner import GameRunnerActor

# Characters to use for evaluation
CHARACTER_POOL = [
    melee.Character.FOX, 
    melee.Character.FALCO,
    melee.Character.MARTH,
    melee.Character.SHEIK,
    melee.Character.JIGGLYPUFF,
    melee.Character.PEACH,
    melee.Character.CPTFALCON,
]

# Fallback agent names if we can't extract from model
DEFAULT_AGENT_NAMES = [
    "Master Player",
    "Ralph",
    "Darkatma",
    "Tempo",
    "xRunRiot",
    "Dragunov",
]

# Define flags for the script
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", None, "Directory containing model files to evaluate", required=True)
flags.DEFINE_string("model_pattern", "*.pkl", "Glob pattern for finding model files")
flags.DEFINE_integer("max_parallel_games", 2, "Maximum number of games to run in parallel")
flags.DEFINE_string("output_dir", "./model_pool_results", "Directory to save results")
flags.DEFINE_integer("max_games", None, "Maximum number of games to run (None for infinite)")
flags.DEFINE_integer("display_interval", 10, "Display TrueSkill rankings every N games")
flags.DEFINE_bool("randomize_characters", True, "Whether to randomize characters each game")
flags.DEFINE_string("dolphin_path", None, "Path to Slippi Dolphin executable", required=True)
flags.DEFINE_string("dolphin_iso", None, "Path to SSBM ISO file", required=True)
flags.DEFINE_bool("dolphin_headless", False, "Run Dolphin in headless mode")
flags.DEFINE_float("novelty_weight", 1.0, "Weight for preferring newer/unknown models (higher = stronger preference)")

class ModelPoolEvaluator:
    """Manages a pool of games between different models."""
    
    def __init__(self, 
                 model_dir: str,
                 model_pattern: str = "*.pkl",
                 output_dir: str = "./model_pool_results",
                 max_parallel_games: int = 2,
                 display_interval: int = 10,
                 randomize_characters: bool = True,
                 novelty_weight: float = 1.0):
        self.model_dir = model_dir
        self.model_pattern = model_pattern
        self.output_dir = output_dir
        self.max_parallel_games = max_parallel_games
        self.display_interval = display_interval
        self.randomize_characters = randomize_characters
        self.novelty_weight = novelty_weight
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Ray if not already started
        if not ray.is_initialized():
            ray.init()
        
        # Initialize ranker
        self.ranker = ModelRanker(output_dir)
        
        # Find all model files in the directory
        model_pattern_path = os.path.join(model_dir, model_pattern)
        self.models = glob.glob(model_pattern_path)
        
        if not self.models:
            raise ValueError(f"No models found matching pattern {model_pattern_path}")
        
        # Ensure all models are registered with the ranker
        self._initialize_model_ratings()
        
        # Debug: inspect the ranker's API
        self._debug_print_ranker_info()
        
        # Process tracking
        self.active_games: Dict[int, Dict] = {}
        self.game_count = 0
        
        # Load model info (including names)
        self.model_info = self._load_model_info()
        
        # Track matchup history - how many times each pair has played
        self.matchup_counts = {}
        # Track the total number of games each model has played
        self.model_games_played = {model: 0 for model in self.models}
        
        # Initialize match counts
        for model1 in self.models:
            for model2 in self.models:
                if model1 <= model2:  # Use alphabetical ordering to avoid duplicates
                    self.matchup_counts[(model1, model2)] = 0
        
        # Load match history from match_records.json
        self._load_match_history()
        
        # Parameters for model selection
        self.min_matchups_per_pair = 2  # Ensure each pair plays at least this many games
        self.exploration_weight = 1.0   # Weight for uncertainty (sigma)
        self.similarity_weight = 2.0    # Weight for rating similarity
        self.diversity_weight = 0.5     # Weight for matchup diversity
        self.initial_temperature = 5.0  # Initial randomness in selection
        self.final_temperature = 0.5    # Final randomness in selection
        self.temperature_decay = 0.01   # How quickly to reduce randomness
        
        logging.info(f"Initialized ModelPoolEvaluator with {len(self.models)} models")
        for i, model in enumerate(self.models, 1):
            model_name = os.path.basename(model)
            agent_names = self.model_info[model].get('agent_names', ['Unknown'])
            games_played = self.model_games_played[model]
            rating_info = self._get_model_rating_info(model)
            logging.info(f"  {i}. {model_name} - Names: {agent_names} - Games: {games_played} - Rating: {rating_info}")
    
    def _initialize_model_ratings(self):
        """Ensure all models have TrueSkill ratings initialized.
        This ensures new models are registered with the ranker."""
        # Get current ratings
        existing_ratings = {rating[0]: True for rating in self.ranker.get_sorted_ratings()}
        
        # Register any new models that don't have ratings yet
        for model in self.models:
            if model not in existing_ratings:
                logging.info(f"Initializing rating for new model: {os.path.basename(model)}")
                # Register the model with the ranker
                self.ranker.ensure_model_registered(model)

    def _load_match_history(self):
        """Load match history from match_records.json and compute matchup counts and game counts."""
        match_records_file = os.path.join(self.output_dir, "match_records.json")
        if os.path.exists(match_records_file):
            try:
                with open(match_records_file, 'r') as f:
                    match_records = json.load(f)
            except json.JSONDecodeError as e:
                # This is a critical error - the file exists but is corrupt
                raise ValueError(f"Failed to parse match_records.json: {e}")
            except Exception as e:
                # Other file reading errors are critical
                raise IOError(f"Failed to read match_records.json: {e}")
            
            # Process each match record to count matchups and games played
            for match in match_records:
                winner = match.get('winner')
                loser = match.get('loser')
                
                # Skip invalid records
                if not winner or not loser:
                    logging.warning(f"Skipping invalid match record missing winner or loser")
                    continue
                
                # Match paths to current model paths as needed
                winner_match = self._match_model_path(winner)
                loser_match = self._match_model_path(loser)
                
                if winner_match and loser_match:
                    # Update matchup count
                    key = (winner_match, loser_match) if winner_match <= loser_match else (loser_match, winner_match)
                    self.matchup_counts[key] = self.matchup_counts.get(key, 0) + 1
                    
                    # Update game counts for both models
                    self.model_games_played[winner_match] = self.model_games_played.get(winner_match, 0) + 1
                    self.model_games_played[loser_match] = self.model_games_played.get(loser_match, 0) + 1
                else:
                    logging.warning(f"Could not match models: {os.path.basename(winner)} or {os.path.basename(loser)}")
            
            logging.info(f"Loaded {len(match_records)} match records from {match_records_file}")
            logging.info(f"Computed matchup counts for {len(self.matchup_counts)} model pairs")
            logging.info(f"Computed game counts for {len(self.model_games_played)} models")
        else:
            logging.info("No previous match records found - starting with fresh history")
    
    def _match_model_path(self, model_path):
        """Match a model path from match records to a current model path.
        
        This handles cases where models might have moved or paths have changed.
        Returns the current model path if a match is found, or None if no match.
        """
        # If the exact path exists in our models, return it
        if model_path in self.models:
            return model_path
        
        # Try matching by basename
        basename = os.path.basename(model_path)
        matching_models = [m for m in self.models if os.path.basename(m) == basename]
        
        if matching_models:
            return matching_models[0]
        
        # No match found
        return None

    def _get_model_rating_info(self, model: str) -> str:
        """Get a string with rating information for a model."""
        # Try to get all ratings from the ranker
        for rating_entry in self.ranker.get_sorted_ratings():
            if isinstance(rating_entry, tuple) and len(rating_entry) >= 4:
                model_path, mu, sigma, conservative_rating = rating_entry
                if model_path == model:
                    return f"μ={mu:.2f}, σ={sigma:.2f}, rating={conservative_rating:.2f}"
        return "Unrated"
    
    def _debug_print_ranker_info(self):
        """Print debug information about the ranker's API and data structure."""
        # Debug the ranker's API
        methods = [name for name in dir(self.ranker) if callable(getattr(self.ranker, name)) and not name.startswith('_')]
        logging.info(f"ModelRanker available methods: {methods}")
        
        # Try to get sorted ratings and inspect their structure
        sorted_ratings = self.ranker.get_sorted_ratings()
        if sorted_ratings:
            logging.info(f"Ranker has {len(sorted_ratings)} ratings")
            if len(sorted_ratings) > 0:
                sample = sorted_ratings[0]
                logging.info(f"Rating format: (model_path, mu, sigma, conservative_rating)")
                if isinstance(sample, tuple) and len(sample) >= 4:
                    model_path, mu, sigma, conservative = sample
                    logging.info(f"  Example: ({os.path.basename(model_path)}, {mu:.2f}, {sigma:.2f}, {conservative:.2f})")
    
    def _load_model_info(self) -> Dict[str, Dict]:
        """Load information about each model, including agent names."""
        model_info = {}
        
        for model_path in self.models:
            try:
                # Try to load the model state to extract names
                state = saving.load_state_from_disk(model_path)
                agent_names = eval_lib.get_name_from_rl_state(state) or ['Unknown']

                logging.debug(f"Agent names for model {model_path}: {agent_names}")
                
                # Store the model info
                model_info[model_path] = {
                    'agent_names': agent_names,
                    'name_map': state.get('name_map', {})
                }
                
                logging.info(f"Loaded model {os.path.basename(model_path)} with names: {agent_names}")
            except Exception as e:
                logging.warning(f"Error loading model {model_path}: {e}")
                model_info[model_path] = {
                    'agent_names': ['Unknown'],
                    'name_map': {}
                }
        
        return model_info
    
    def _get_random_characters(self) -> List[melee.Character]:
        """Get random characters for a game."""
        return random.choices(CHARACTER_POOL, k=4)
    
    def _get_agent_names(self, model_path: str) -> List[str]:
        """Get agent names for a model, preferring those from the model config."""
        model_info = self.model_info.get(model_path, {})
        agent_names = model_info.get('agent_names', [])

        logging.debug(f"Getting agent names for {model_path}: {agent_names}")
        
        # If we have valid agent names, return them
        if agent_names and agent_names[0] != 'Unknown':
            # If we have only one name but need more, repeat it
            if len(agent_names) == 1:
                return agent_names * 2  # We need 2 agents per model
            # If we have multiple names, choose 2 random ones with potential duplicates
            elif len(agent_names) > 1:
                # Use random.choices instead of random.sample to allow duplicates
                return random.choices(agent_names, k=2)
            # If we have exactly 2 names, use them
            return agent_names
            
        # Otherwise fall back to random names
        # Use random.choices instead of random.sample to allow duplicates
        return random.choices(DEFAULT_AGENT_NAMES, k=2)
    
    def _update_matchup_count(self, model1: str, model2: str) -> None:
        """Update the count of matches between two models."""
        # Ensure consistent key ordering
        key = (model1, model2) if model1 <= model2 else (model2, model1)
        self.matchup_counts[key] = self.matchup_counts.get(key, 0) + 1
    
    def _get_matchup_count(self, model1: str, model2: str) -> int:
        """Get the count of matches between two models."""
        # Ensure consistent key ordering
        key = (model1, model2) if model1 <= model2 else (model2, model1)
        return self.matchup_counts.get(key, 0)
    
    def _get_current_temperature(self) -> float:
        """Calculate the current temperature for model selection.
        
        Temperature decreases as more games are played, transitioning from
        exploration to exploitation.
        """
        # Calculate decay factor based on number of games played
        decay_factor = 1.0 - min(1.0, self.game_count * self.temperature_decay)
        
        # Interpolate between initial and final temperature
        return self.initial_temperature * decay_factor + self.final_temperature * (1.0 - decay_factor)
    
    def _get_model_uncertainty(self, model: str) -> float:
        """Get the uncertainty (sigma) of a model's TrueSkill rating."""
        try:
            # Try to get all ratings from the ranker
            sorted_ratings = self.ranker.get_sorted_ratings()
            
            # From trueskill_ranker.py we can see the format is:
            # List[Tuple[str, float, float, float]] = (model_name, mu, sigma, conservative_rating)
            for rating_entry in sorted_ratings:
                if isinstance(rating_entry, tuple) and len(rating_entry) >= 3:
                    model_path, mu, sigma, *_ = rating_entry
                    if model_path == model:
                        return sigma
            
            # If model not found, return default value
            return 25.0  # Default initial sigma in TrueSkill
        except Exception as e:
            logging.warning(f"Error getting model uncertainty: {e}")
            return 25.0  # Default value on error
    
    def _get_model_rating(self, model: str) -> float:
        """Get the conservative TrueSkill rating for a model."""
        try:
            # Try to get all ratings from the ranker
            sorted_ratings = self.ranker.get_sorted_ratings()
            
            # From trueskill_ranker.py we can see the format is:
            # List[Tuple[str, float, float, float]] = (model_name, mu, sigma, conservative_rating)
            for rating_entry in sorted_ratings:
                if isinstance(rating_entry, tuple) and len(rating_entry) >= 4:
                    model_path, mu, sigma, conservative_rating = rating_entry
                    if model_path == model:
                        return conservative_rating
            
            # If model not found, return default value
            return 25.0  # Default initial mu in TrueSkill
        except Exception as e:
            logging.warning(f"Error getting model rating: {e}")
            return 25.0  # Default value on error
    
    def _calculate_rating_similarity(self, model1: str, model2: str) -> float:
        """Calculate how similar the ratings of two models are.
        
        Returns a value between 0 and 1, where 1 means identical ratings.
        """
        rating1 = self._get_model_rating(model1)
        rating2 = self._get_model_rating(model2)
        
        # Calculate similarity - highest when ratings are close
        difference = abs(rating1 - rating2)
        similarity = 1.0 / (1.0 + difference / 5.0)  # Normalize with scale factor
        
        return similarity
    
    def _get_model_novelty_score(self, model: str) -> float:
        """Calculate a novelty score for a model based on number of games played.
        
        Returns a value between 0 and 1, where 1 represents a completely new model
        and values approach 0 as the model plays more games.
        """
        games_played = self.model_games_played.get(model, 0)
        # Exponential decay function: higher for fewer games played
        return math.exp(-0.1 * games_played)
    
    def _calculate_matchup_score(self, model1: str, model2: str) -> float:
        """Calculate a score for this matchup based on exploration/exploitation criteria."""
        try:
            # If we have fewer than the minimum required matches, prioritize this matchup
            matchup_count = self._get_matchup_count(model1, model2)
            if matchup_count < self.min_matchups_per_pair:
                return 1000.0 - matchup_count  # Very high score to ensure this pair gets picked
            
            # Get uncertainties (sigmas) for both models
            uncertainty1 = self._get_model_uncertainty(model1)
            uncertainty2 = self._get_model_uncertainty(model2)
            
            # Get rating similarity (higher is better)
            rating_similarity = self._calculate_rating_similarity(model1, model2)
            
            # Calculate matchup diversity score (higher for less-played matchups)
            diversity_score = 1.0 / (1.0 + matchup_count)
            
            # Calculate novelty scores for both models (higher for newer models)
            novelty1 = self._get_model_novelty_score(model1)
            novelty2 = self._get_model_novelty_score(model2)
            novelty_score = (novelty1 + novelty2) / 2
            
            # Calculate final score as weighted sum of components
            score = (
                self.exploration_weight * (uncertainty1 + uncertainty2) +
                self.similarity_weight * rating_similarity +
                self.diversity_weight * diversity_score +
                self.novelty_weight * novelty_score
            )
            
            return score
        except Exception as e:
            # Log error and return a default score based on matchup count
            logging.warning(f"Error calculating matchup score for {os.path.basename(model1)} vs {os.path.basename(model2)}: {e}")
            # Fallback to a simple diversity-based score
            matchup_count = self._get_matchup_count(model1, model2)
            return 100.0 / (1.0 + matchup_count)  # Simple fallback score
    
    def _select_models_for_game(self) -> Tuple[str, str]:
        """Select two models to play against each other using an intelligent strategy.
        
        This balances exploration (high uncertainty models) and exploitation 
        (similar-rated models) to maximize information gain from each game.
        """
        # If we have fewer than 2 models, duplicate the single model
        if len(self.models) == 1:
            return self.models[0], self.models[0]
        
        # Get current temperature for controlling exploration vs. exploitation
        temperature = self._get_current_temperature()
        
        # Calculate scores for all possible matchups
        matchup_scores = {}
        for i, model1 in enumerate(self.models):
            for model2 in self.models[i+1:]:  # Avoid duplicate matchups
                if model1 != model2:  # Skip self-matchups
                    score = self._calculate_matchup_score(model1, model2)
                    matchup_scores[(model1, model2)] = score
        
        #logging.info(f"Matchup scores: {matchup_scores}")
        
        # Convert scores to probabilities using softmax with temperature
        max_score = max(matchup_scores.values()) if matchup_scores else 0
        exp_scores = {
            matchup: math.exp((score - max_score) / temperature) 
            for matchup, score in matchup_scores.items()
        }
        total_exp_score = sum(exp_scores.values())
        
        if total_exp_score == 0:
            # Fallback to random selection if all scores are extremely low
            return tuple(random.sample(self.models, 2))
        
        # Create probability distribution
        matchup_probs = {
            matchup: score / total_exp_score
            for matchup, score in exp_scores.items()
        }
        
        # Select a matchup based on probabilities
        matchups = list(matchup_probs.keys())
        probs = list(matchup_probs.values())
        
        try:
            selected_matchup = random.choices(matchups, weights=probs, k=1)[0]
            
            # Log selection details for debugging
            logging.debug(f"Selected matchup {[os.path.basename(m) for m in selected_matchup]}")
            logging.debug(f"  Score: {matchup_scores[selected_matchup]:.2f}")
            logging.debug(f"  Probability: {matchup_probs[selected_matchup]:.4f}")
            logging.debug(f"  Temperature: {temperature:.2f}")
            
            # Update the matchup count for next time
            self._update_matchup_count(selected_matchup[0], selected_matchup[1])
            
            return selected_matchup
            
        except (IndexError, ValueError) as e:
            # Fallback to random selection if there's an error
            logging.warning(f"Error in model selection: {e}. Falling back to random selection.")
            selected_models = tuple(random.sample(self.models, 2))
            self._update_matchup_count(selected_models[0], selected_models[1])
            return selected_models
    
    def _create_game_config(self, game_id: int) -> Dict:
        """Create configuration for a new game."""
        model1, model2 = self._select_models_for_game()
        
        # Assign characters
        characters = self._get_random_characters() if self.randomize_characters else [
            melee.Character.FOX, melee.Character.MARTH, melee.Character.FALCO, melee.Character.SHEIK
        ]
        
        # Get agent names from models
        model1_names = self._get_agent_names(model1)
        model2_names = self._get_agent_names(model2)
        
        # Team 1: player 1 and 4 (model1)
        # Team 2: player 2 and 3 (model2)
        agent_names = [
            model1_names[0],  # p1 (model1)
            model2_names[0],  # p2 (model2)
            model2_names[-1], # p3 (model2)
            model1_names[-1]  # p4 (model1)
        ]
        
        # Log the game configuration
        logging.info(f"Game {game_id} configuration:")
        logging.info(f"  Model 1 (team 1): {os.path.basename(model1)}")
        logging.info(f"  Model 2 (team 2): {os.path.basename(model2)}")
        logging.info(f"  Characters: {[c.name for c in characters]}")
        logging.info(f"  Agent names: {agent_names}")
        
        return {
            "id": game_id,
            "model1": model1,
            "model2": model2,
            "characters": [c.name for c in characters],
            "agent_names": agent_names,
            "start_time": time.time(),
            "actor": None,
            "future": None,
            "status": "pending"
        }
    
    def _launch_game(self, game_config: Dict) -> None:
        """Launch a new game with the given configuration using a Ray actor."""
        # Create a new GameRunnerActor
        actor = GameRunnerActor.remote()
        
        # Launch the game by calling run_game and get a future
        future = actor.run_game.remote(
            game_id=game_config["id"],
            model1_path=game_config["model1"],
            model2_path=game_config["model2"],
            characters=game_config["characters"],
            agent_names=game_config["agent_names"],
            dolphin_path=FLAGS.dolphin_path,
            dolphin_iso=FLAGS.dolphin_iso,
            dolphin_headless=FLAGS.dolphin_headless
        )
        
        # Update game config
        game_config["actor"] = actor
        game_config["future"] = future
        game_config["status"] = "running"
        
        logging.info(f"Game {game_config['id']} launched")
    
    def _check_game_results(self, game_config: Dict) -> Optional[Dict]:
        """Check if a game has finished and get the results."""
        # If the game doesn't have a future, it wasn't launched properly
        if not game_config.get("future"):
            return None
        
        # Check if the future is ready
        ready, _ = ray.wait([game_config["future"]], timeout=0)
        if not ready:
            return None
        
        # Get the result from the future
        result = ray.get(game_config["future"])
        
        # Check if the game completed successfully
        if result.get("status") != "completed":
            logging.warning(f"Game {game_config['id']} failed: {result.get('error', 'Unknown error')}")
            return None
        
        # Extract relevant information from the result
        return {
            "game_id": game_config["id"],
            "winner": result["model1"] if result["winner_team"] == 0 else result["model2"],
            "loser": result["model2"] if result["winner_team"] == 0 else result["model1"],
            "winner_chars": result["winner_chars"],
            "loser_chars": result["loser_chars"],
            "winner_names": result["winner_names"],
            "loser_names": result["loser_names"],
            "duration": result["duration"],
            "timed_out": result.get("timed_out", False)
        }
    
    def _cleanup_game(self, game_id: int):
        """Clean up resources for a finished game."""
        if game_id not in self.active_games:
            return
            
        game_config = self.active_games[game_id]
        
        # Remove from active games
        del self.active_games[game_id]
    
    def run(self):
        """Run the model pool evaluation."""
        try:
            while True:
                # Check if we've reached the maximum number of games
                if FLAGS.max_games and self.game_count >= FLAGS.max_games:
                    logging.info(f"Reached maximum number of games ({FLAGS.max_games}). Stopping.")
                    break
                
                # Check results of active games
                for game_id in list(self.active_games.keys()):
                    game_config = self.active_games[game_id]
                    
                    # Check if the game has completed
                    results = self._check_game_results(game_config)
                    if results:
                        # Update TrueSkill ratings
                        self.ranker.update_rating(
                            winner_model=results["winner"],
                            loser_model=results["loser"],
                            winner_chars=results["winner_chars"],
                            loser_chars=results["loser_chars"],
                            winner_names=results["winner_names"],
                            loser_names=results["loser_names"]
                        )
                        
                        # Update matchup count for tracking
                        self._update_matchup_count(results["winner"], results["loser"])
                        
                        # Update games played count for both models
                        self.model_games_played[results["winner"]] = self.model_games_played.get(results["winner"], 0) + 1
                        self.model_games_played[results["loser"]] = self.model_games_played.get(results["loser"], 0) + 1
                        
                        self.game_count += 1
                        logging.info(f"Game {game_id} completed: "
                                     f"{os.path.basename(results['winner'])} defeated "
                                     f"{os.path.basename(results['loser'])}")
                        
                        # Display rankings periodically
                        if self.game_count % self.display_interval == 0:
                            self.ranker.display_rankings()
                            
                            # Also log current model selection parameters
                            logging.info(f"Current model selection parameters:")
                            logging.info(f"  Temperature: {self._get_current_temperature():.2f}")
                            logging.info(f"  Games played: {self.game_count}")
                            
                            # Log matchup counts for the top models
                            try:
                                top_models = self._get_top_models(5)
                                if top_models:
                                    logging.info("Matchup counts for top models:")
                                    for i, model1 in enumerate(top_models):
                                        for model2 in top_models[i+1:]:
                                            count = self._get_matchup_count(model1, model2)
                                            logging.info(f"  {os.path.basename(model1)} vs {os.path.basename(model2)}: {count}")
                            except Exception as e:
                                # If there's any error, just log all matchups
                                logging.info(f"Could not get top model matchups: {e}")
                                logging.info("Matchup counts for all models:")
                                matchup_sample = list(self.matchup_counts.items())[:10]  # Show only first 10
                                for (model1, model2), count in matchup_sample:
                                    logging.info(f"  {os.path.basename(model1)} vs {os.path.basename(model2)}: {count}")
                                if len(self.matchup_counts) > 10:
                                    logging.info(f"  ... and {len(self.matchup_counts) - 10} more matchups")
                        
                        # Clean up this game
                        self._cleanup_game(game_id)
                
                # Launch new games if we have capacity
                while len(self.active_games) < self.max_parallel_games:
                    game_id = self.game_count + len(self.active_games) + 1
                    game_config = self._create_game_config(game_id)
                    self._launch_game(game_config)
                    self.active_games[game_id] = game_config
                
                # Sleep to avoid high CPU usage
                time.sleep(5)
                
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Cleaning up...")
        finally:
            # Clean up all active games
            for game_id in list(self.active_games.keys()):
                self._cleanup_game(game_id)
            
            # We don't need to save matchup statistics since they're derived from match_records.json
            
            # Display final rankings
            self.ranker.display_rankings()
            
            # Shut down Ray
            ray.shutdown()
    
    def _get_top_models(self, n: int = 5) -> List[str]:
        """Get the top N models by TrueSkill rating."""
        try:
            # Get sorted ratings from the ranker
            sorted_ratings = self.ranker.get_sorted_ratings()
            
            # From trueskill_ranker.py we can see the format is:
            # List[Tuple[str, float, float, float]] = (model_name, mu, sigma, conservative_rating)
            # Already sorted by conservative rating (highest first)
            top_models = []
            for rating_entry in sorted_ratings[:n]:
                if isinstance(rating_entry, tuple) and len(rating_entry) >= 1:
                    top_models.append(rating_entry[0])
            
            # If we couldn't extract enough models, add some from our full list
            if len(top_models) < n:
                remaining = min(n - len(top_models), len(self.models))
                remaining_models = [m for m in self.models if m not in top_models]
                top_models.extend(remaining_models[:remaining])
            
            return top_models
        except Exception as e:
            logging.debug(f"Error getting top models: {e}")
            return self.models[:min(n, len(self.models))]  # Default to first N models

def main(_):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_dir, "model_pool_eval.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create the evaluator
    evaluator = ModelPoolEvaluator(
        model_dir=FLAGS.model_dir,
        model_pattern=FLAGS.model_pattern,
        output_dir=FLAGS.output_dir,
        max_parallel_games=FLAGS.max_parallel_games,
        display_interval=FLAGS.display_interval,
        randomize_characters=FLAGS.randomize_characters,
        novelty_weight=FLAGS.novelty_weight
    )
    
    # Run the evaluation
    evaluator.run()

if __name__ == '__main__':
    # https://github.com/python/cpython/issues/87115
    __spec__ = None
    app.run(main) 