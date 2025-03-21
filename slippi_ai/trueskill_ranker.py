"""TrueSkill rating system for model pool evaluation."""

import json
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# We'll use the trueskill package - note this will need to be installed
try:
    import trueskill
except ImportError:
    raise ImportError(
        "The trueskill package is required. Please install it with: pip install trueskill"
    )

class ModelRanker:
    """Manages TrueSkill ratings for a pool of models."""
    
    def __init__(self, output_dir: str = "./model_pool_results"):
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize ratings storage
        self.ratings: Dict[str, trueskill.Rating] = {}
        self.matches: List[Dict] = []
        
        # Setup trueskill environment
        trueskill.setup(draw_probability=0.0)  # No draws in SSBM
        
        # Load existing ratings if available
        self.ratings_file = os.path.join(output_dir, "trueskill_ratings.json")
        self.matches_file = os.path.join(output_dir, "match_records.json")
        self._load_ratings()
        self._load_matches()
    
    def _load_ratings(self):
        """Load existing ratings from file if available."""
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, 'r') as f:
                    ratings_data = json.load(f)
                
                for model_name, rating_data in ratings_data.items():
                    self.ratings[model_name] = trueskill.Rating(
                        mu=rating_data['mu'],
                        sigma=rating_data['sigma']
                    )
                print(f"Loaded ratings for {len(self.ratings)} models")
            except Exception as e:
                print(f"Error loading ratings: {e}")
    
    def _load_matches(self):
        """Load existing match records from file if available."""
        if os.path.exists(self.matches_file):
            try:
                with open(self.matches_file, 'r') as f:
                    self.matches = json.load(f)
                print(f"Loaded {len(self.matches)} match records")
            except Exception as e:
                print(f"Error loading match records: {e}")
    
    def save_ratings(self):
        """Save current ratings to file."""
        ratings_data = {}
        for model_name, rating in self.ratings.items():
            ratings_data[model_name] = {
                'mu': rating.mu,
                'sigma': rating.sigma,
                'conservative_rating': rating.mu - 3 * rating.sigma
            }
        
        with open(self.ratings_file, 'w') as f:
            json.dump(ratings_data, f, indent=2)
    
    def save_matches(self):
        """Save match records to file."""
        with open(self.matches_file, 'w') as f:
            json.dump(self.matches, f, indent=2)
    
    def get_rating(self, model_name: str) -> trueskill.Rating:
        """Get rating for a model, creating a new one if it doesn't exist."""
        if model_name not in self.ratings:
            self.ratings[model_name] = trueskill.Rating()
        return self.ratings[model_name]
    
    def update_rating(self, 
                     winner_model: str, 
                     loser_model: str, 
                     winner_chars: List[str], 
                     loser_chars: List[str],
                     winner_names: List[str],
                     loser_names: List[str]):
        """Update ratings after a match."""
        # Get current ratings
        winner_rating = self.get_rating(winner_model)
        loser_rating = self.get_rating(loser_model)
        
        # Update ratings
        winner_new_rating, loser_new_rating = trueskill.rate_1vs1(winner_rating, loser_rating)
        
        # Store new ratings
        self.ratings[winner_model] = winner_new_rating
        self.ratings[loser_model] = loser_new_rating
        
        # Record match
        match_record = {
            'timestamp': time.time(),
            'winner': winner_model,
            'loser': loser_model,
            'winner_chars': winner_chars,
            'loser_chars': loser_chars,
            'winner_names': winner_names,
            'loser_names': loser_names,
            'winner_rating_before': {
                'mu': float(winner_rating.mu),
                'sigma': float(winner_rating.sigma)
            },
            'winner_rating_after': {
                'mu': float(winner_new_rating.mu),
                'sigma': float(winner_new_rating.sigma)
            },
            'loser_rating_before': {
                'mu': float(loser_rating.mu),
                'sigma': float(loser_rating.sigma)
            },
            'loser_rating_after': {
                'mu': float(loser_new_rating.mu),
                'sigma': float(loser_new_rating.sigma)
            }
        }
        
        self.matches.append(match_record)
        
        # Save updated data
        self.save_ratings()
        self.save_matches()
        
        return winner_new_rating, loser_new_rating
    
    def get_sorted_ratings(self) -> List[Tuple[str, float, float, float]]:
        """Get sorted list of models by rating."""
        if not self.ratings:
            return []
            
        # Sort by conservative rating (mu - 3*sigma)
        sorted_models = []
        for model_name, rating in self.ratings.items():
            conservative_rating = rating.mu - 3 * rating.sigma
            sorted_models.append((model_name, rating.mu, rating.sigma, conservative_rating))
        
        return sorted(sorted_models, key=lambda x: x[3], reverse=True)
    
    def display_rankings(self):
        """Print current rankings."""
        sorted_ratings = self.get_sorted_ratings()
        
        if not sorted_ratings:
            print("No ratings available yet.")
            return
        
        print("\n=== Model Rankings (by TrueSkill) ===")
        print(f"{'Rank':<5} {'Model':<30} {'Rating':<12} {'Uncertainty':<12} {'Conservative':<12}")
        print("-" * 75)
        
        for i, (model_name, mu, sigma, conservative) in enumerate(sorted_ratings, 1):
            display_name = os.path.basename(model_name)
            print(f"{i:<5} {display_name:<30} {mu:<5.2f}±{sigma:<6.2f} {sigma:<12.2f} {conservative:<12.2f}") 