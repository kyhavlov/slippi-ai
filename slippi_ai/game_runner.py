"""Game runner for evaluating AI models in SSBM doubles matches.

This module provides a Ray actor for running doubles games between different AI models
and determining the winner based on game state.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import melee
import ray

from slippi_ai import eval_lib, utils, saving
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.rl import run_lib

@ray.remote
class GameRunnerActor:
    """Ray actor for running doubles games between models.

    This actor handles:
    - Loading models and setting up Dolphin
    - Running a complete game
    - Determining the winner
    - Returning structured result data
    """

    def __init__(self):
        """Initialize the game runner actor."""
        self.dolphin = None
        self.agents = []
        self.start_time = None
        self.game_id = None
        self.model1_path = None  # Store model paths
        self.model2_path = None  # Store model paths
        self.agent_names = []
        self.character_names = []
        eval_lib.disable_gpus()

    def run_game(self, 
                game_id: int,
                model1_path: str, 
                model2_path: str, 
                characters: List[str],
                agent_names: List[str],
                dolphin_path: str,
                dolphin_iso: str,
                dolphin_headless: bool = False) -> Dict[str, Any]:
        """Run a doubles game between two models.

        Args:
            game_id: Unique identifier for this game
            model1_path: Path to first model (team 1: ports 1 and 4)
            model2_path: Path to second model (team 2: ports 2 and 3)
            characters: List of character names for each port [p1, p2, p3, p4]
            agent_names: List of agent names for each port [p1, p2, p3, p4]
            dolphin_path: Path to Dolphin executable
            dolphin_iso: Path to SSBM ISO
            dolphin_headless: Whether to run Dolphin in headless mode

        Returns:
            Dict containing game results including winner, stats, etc.
        """
        self.game_id = game_id
        self.start_time = time.time()
        
        # Store model paths and agent names for later use
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.agent_names = agent_names
        # Store character names for later reference
        self.character_names = characters
        
        # Log the input parameters
        logging.info(f"Game {self.game_id} - Starting game with characters: {characters}")
        logging.info(f"Game {self.game_id} - Agent names: {agent_names}")

        try:
            # Set up players for each port
            player_configs = self._create_player_configs(
                model1_path, model2_path, characters, agent_names)
            
            # Set up Dolphin
            dolphin, agents = self._setup_dolphin_and_agents(
                player_configs, dolphin_path, dolphin_iso, dolphin_headless)
            
            self.dolphin = dolphin
            self.agents = agents
            
            # Run the game and get results
            result = self._run_game_loop(dolphin, agents)

            # wait for game to transition to menu
            while dolphin.next_gamestate().menu_state == melee.Menu.IN_GAME:
                time.sleep(0.01)
            
            time.sleep(1)
            
            # Add metadata to result
            result.update({
                "game_id": game_id,
                "duration": time.time() - self.start_time,
                "model1": model1_path,
                "model2": model2_path,
                "characters": characters,
                "agent_names": agent_names
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Error in game {game_id}: {str(e)}")
            return {
                "game_id": game_id,
                "error": str(e),
                "status": "failed",
                "duration": time.time() - self.start_time
            }
        finally:
            self._cleanup()
    
    def _create_player_configs(self, 
                              model1_path: str, 
                              model2_path: str,
                              characters: List[str],
                              agent_names: List[str]) -> Dict[int, Dict]:
        """Create player configurations for all ports."""
        melee_characters = [getattr(melee.Character, char) for char in characters]
        
        # Define player config for each port
        player_configs = {}
        for port in range(1, 5):
            # Team 1: ports 1 and 4, Team 2: ports 2 and 3
            is_team1 = port == 1 or port == 4
            model_path = model1_path if is_team1 else model2_path
            idx = port - 1  # 0-indexed
            
            player_configs[port] = {
                "type": "ai",
                "character": melee_characters[idx],
                "level": 9,  # Required by get_player but not used for AI players
                "ai": {
                    "path": model_path,
                    "name": agent_names[idx],
                    "async_inference": True
                }
            }
        
        return player_configs
    
    def _setup_dolphin_and_agents(self,
                                 player_configs: Dict[int, Dict],
                                 dolphin_path: str,
                                 dolphin_iso: str,
                                 dolphin_headless: bool) -> Tuple[dolphin_lib.Dolphin, List[eval_lib.Agent]]:
        """Set up Dolphin and AI agents."""
        # Create player objects from configs
        players = {
            port: eval_lib.get_player(**config)
            for port, config in player_configs.items()
        }
        
        # Initialize agents
        agents = []
        for port, teammate_port in zip((1, 2, 3, 4), (4, 3, 2, 1)):
            player = players[port]
            if isinstance(player, dolphin_lib.AI):
                # Set up team-specific opponent ports
                if port == 1 or port == 4:  # Team 1
                    opponent_port = 2
                else:  # Team 2
                    opponent_port = 1
                
                agent = eval_lib.build_agent(
                    port=port,
                    teammate_port=teammate_port,
                    opponent_port=opponent_port,
                    console_delay=0,  # No delay for evaluation
                    **player_configs[port]["ai"]
                )
                agent.start()
                agents.append(agent)
                
                # Update character based on agent config
                eval_lib.update_character(player, agent.config)
        
        # Initialize Dolphin
        replay_dir = self.game_id % 2
        dolphin_config = {
            "path": dolphin_path,
            "iso": dolphin_iso,
            "headless": dolphin_headless,
            "save_replays": True,
            "replay_dir": f"eval/replays/{replay_dir}",
            "infinite_time": False,
            "disable_audio": True,
            "blocking_input": True,
            "fullscreen": False,
            "emulation_speed": 0.0,
            "stage": melee.Stage.RANDOM_STAGE,
        }
        
        dolphin = dolphin_lib.Dolphin(
            players=players,
            desired_teams={1: 0, 2: 1, 3: 1, 4: 0},  # Team assignments
            **dolphin_config
        )
        
        # Connect controllers to agents
        for agent in agents:
            agent.set_controller(dolphin.controllers[agent._port])
        
        return dolphin, agents
    
    def _run_game_loop(self, dolphin: dolphin_lib.Dolphin, agents: List[eval_lib.Agent]) -> Dict:
        """Run the game loop and determine the winner."""
        step_timer = utils.Profiler()
        game_start_time = time.time()
        game_stats = {"frames": 0}
        team_stocks = {0: 8, 1: 8}  # Each team starts with 8 stocks (4 per player)
        
        try:
            # Main game loop
            logging.info(f"Game {self.game_id} - Starting game loop")
            while True:
                try:
                    gamestate = dolphin.step()
                    game_stats["frames"] += 1
                    
                    # Process agent steps
                    with step_timer:
                        for agent in agents:
                            agent.step(gamestate)
                    
                    # Log performance periodically
                    if gamestate.frame > 0 and gamestate.frame % (5 * 60) == 0:
                        logging.info(f'Game {self.game_id} - Frame {gamestate.frame}, step_time: {step_timer.mean_time():.3f}')
                    
                    # Check game status - are we in-game?
                    if gamestate.menu_state != melee.Menu.IN_GAME:
                        continue
                    
                    # Track stock counts for each team
                    self._update_team_stocks(gamestate, team_stocks)
                    
                    # Check if game has ended (one team has lost all stocks)
                    winner_team = self._check_game_end(team_stocks)
                    if winner_team is not None:
                        logging.info(f"Game {self.game_id} completed after {game_stats['frames']} frames")
                        logging.info(f"Game {self.game_id} - Winning team: {winner_team} with stocks remaining: {team_stocks[winner_team]}")
                        return self._create_result_dict(winner_team, game_stats, gamestate)
                    
                    # Safety check - game is taking too long
                    if time.time() - game_start_time > 600:  # 10 minute limit
                        logging.warning(f"Game {self.game_id} timed out after 10 minutes")
                        # Determine winner based on current stocks
                        winner_team = 0 if team_stocks[0] > team_stocks[1] else 1
                        return self._create_result_dict(winner_team, game_stats, gamestate, timed_out=True)
                
                except Exception as e:
                    # Handle errors within the game loop
                    logging.error(f"Game {self.game_id} - Error in frame: {game_stats['frames']}: {str(e)}")
                    # If we've been running for a while, try to determine a winner and exit
                    if game_stats['frames'] > 600:  # More than 10 seconds of gameplay
                        winner_team = 0 if team_stocks[0] > team_stocks[1] else 1
                        logging.warning(f"Game {self.game_id} - Ending due to error, declaring team {winner_team} the winner")
                        try:
                            return self._create_result_dict(winner_team, game_stats, gamestate, timed_out=True)
                        except:
                            # Last resort, create a minimal result
                            return {
                                "status": "completed",
                                "error_recovery": True,
                                "winner_team": winner_team,
                                "winner_model": self.model1_path if winner_team == 0 else self.model2_path,
                                "loser_model": self.model2_path if winner_team == 0 else self.model1_path,
                                "game_stats": game_stats
                            }
                    else:
                        # If early in the game, re-raise the error
                        raise
                
        except Exception as e:
            logging.error(f"Error in game loop for game {self.game_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "game_stats": game_stats
            }
    
    def _update_team_stocks(self, gamestate: melee.GameState, team_stocks: Dict[int, int]):
        """Update the stock counts for each team."""
        if not hasattr(gamestate, "players") or not gamestate.players:
            return
        
        # Reset stock counts
        new_team_stocks = {0: 0, 1: 0}
        
        # Count stocks for each team
        try:
            for port, player in gamestate.players.items():
                if port in (1, 4):  # Team 1
                    new_team_stocks[0] += player.stock if hasattr(player, "stock") else 0
                elif port in (2, 3):  # Team 2
                    new_team_stocks[1] += player.stock if hasattr(player, "stock") else 0
            
            # Update team stocks
            team_stocks[0] = new_team_stocks[0]
            team_stocks[1] = new_team_stocks[1]
            
            # Log stock counts periodically
            if gamestate.frame % 600 == 0:  # Log every 10 seconds (@ 60fps)
                logging.info(f"Game {self.game_id} - Team 1 stocks: {team_stocks[0]}, Team 2 stocks: {team_stocks[1]}")
        except Exception as e:
            logging.warning(f"Error updating team stocks: {e}")
            # Don't update team stocks on error
    
    def _check_game_end(self, team_stocks: Dict[int, int]) -> Optional[int]:
        """Check if the game has ended by one team losing all stocks.
        
        Returns:
            The winning team index (0 or 1) or None if game is still ongoing.
        """
        if team_stocks[0] == 0:
            return 1  # Team 2 wins
        elif team_stocks[1] == 0:
            return 0  # Team 1 wins
        return None  # Game still ongoing
    
    def _create_result_dict(self, winner_team: int, game_stats: Dict, 
                           gamestate: melee.GameState, timed_out: bool = False) -> Dict:
        """Create the final result dictionary."""
        # Use the original model paths that were stored in run_game
        model1_path = self.model1_path
        model2_path = self.model2_path
        
        # Log the paths for debugging
        logging.info(f"Game {self.game_id} - Model 1 path: {model1_path}")
        logging.info(f"Game {self.game_id} - Model 2 path: {model2_path}")
        
        # Use the actual agent names that were passed into run_game
        if hasattr(self, 'agent_names') and len(self.agent_names) == 4:
            # Team 1 is player 1 and 4
            team1_names = [self.agent_names[0], self.agent_names[3]]
            # Team 2 is player 2 and 3
            team2_names = [self.agent_names[1], self.agent_names[2]]
        else:
            # Fallback to default names if agent_names is not available
            team1_names = ["Team 1 P1", "Team 1 P4"]
            team2_names = ["Team 2 P2", "Team 2 P3"]
        
        # Define the ports for each team
        team1_ports = [1, 4]  # Team 1 ports
        team2_ports = [2, 3]  # Team 2 ports
        
        # Determine winner and loser ports
        winner_ports = team1_ports if winner_team == 0 else team2_ports
        loser_ports = team2_ports if winner_team == 0 else team1_ports
        
        # Use character configuration directly instead of trying to extract from gamestate
        # This ensures we never need to reconstruct character information
        if hasattr(self, 'character_names') and len(self.character_names) == 4:
            # Directly map the original character names to winner and loser teams
            # based on the winner_team value
            if winner_team == 0:  # Team 1 won (ports 1 and 4)
                # Winner is team 1 (ports 1 and 4)
                winner_chars = [self.character_names[0], self.character_names[3]]
                # Loser is team 2 (ports 2 and 3)
                loser_chars = [self.character_names[1], self.character_names[2]]
            else:  # Team 2 won (ports 2 and 3)
                # Winner is team 2 (ports 2 and 3)
                winner_chars = [self.character_names[1], self.character_names[2]]
                # Loser is team 1 (ports 1 and 4)
                loser_chars = [self.character_names[0], self.character_names[3]]
                
            logging.info(f"Game {self.game_id} - Using original character configuration")
            logging.info(f"Game {self.game_id} - Winner characters: {winner_chars}")
            logging.info(f"Game {self.game_id} - Loser characters: {loser_chars}")
        else:
            # Fallback if character_names is not available (should never happen)
            logging.warning(f"Game {self.game_id} - No character_names available, using fallback values")
            winner_chars = ["FOX", "MARTH"]
            loser_chars = ["FALCO", "SHEIK"]
        
        return {
            "status": "completed",
            "winner_team": winner_team,
            "winner_model": model1_path if winner_team == 0 else model2_path,
            "loser_model": model2_path if winner_team == 0 else model1_path,
            "winner_chars": winner_chars,
            "loser_chars": loser_chars,
            "winner_names": team1_names if winner_team == 0 else team2_names,
            "loser_names": team2_names if winner_team == 0 else team1_names,
            "timed_out": timed_out,
            "game_stats": game_stats
        }
    
    def _cleanup(self):
        """Clean up resources."""
        if self.agents:
            for agent in self.agents:
                try:
                    agent.stop()
                except:
                    pass
            self.agents = []
        
        if self.dolphin:
            try:
                self.dolphin.stop()
            except:
                pass
            self.dolphin = None 