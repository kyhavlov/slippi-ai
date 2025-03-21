from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

import melee

from slippi_ai.types import (
  GAME_TYPE,
  LIBMELEE_BUTTONS,
  Buttons,
  Controller,
  Game,
  InvalidGameError,
  Player,
  Stick,
  nt_to_nest,
)

def get_stick(stick: Tuple[float]) -> Stick:
  return Stick(*map(np.float32, stick))

def get_buttons(button: Dict[melee.Button, bool]) -> Buttons:
  return Buttons(**{
      name: button[lm_button]
      for name, lm_button in LIBMELEE_BUTTONS.items()
  })

def get_controller(cs: melee.ControllerState) -> Controller:
  return Controller(
      main_stick=get_stick(cs.main_stick),
      c_stick=get_stick(cs.c_stick),
      shoulder=cs.l_shoulder,
      buttons=get_buttons(cs.button),
  )

def get_player(player: melee.PlayerState) -> Player:
  if player.action == melee.Action.UNKNOWN_ANIMATION:
    raise InvalidGameError('UNKNOWN_ANIMATION')

  return Player(
      percent=player.percent,
      facing=player.facing,
      x=player.position.x,
      y=player.position.y,
      action=player.action.value,
      character=player.character.value,
      jumps_left=player.jumps_left,
      shield_strength=player.shield_strength,
      on_ground=player.on_ground,
      is_dead=player.stock == 0,
      stocks_left=player.stock,
      controller=get_controller(player.controller_state),
      # v2.1.0
      invulnerable=player.invulnerable,
      # v3.5.0
      # player.speed_air_x_self,
      # player.speed_ground_x_self,
      # player.speed_x_attack,
      # player.speed_y_attack,
      # player.speed_y_self,
  )

def get_game(
    game: melee.GameState,
    ports: Optional[Sequence[int]] = None,
    singles_opponent_port: int = 2,
) -> Game:
  ports = ports or sorted(game.players)
  
  assert singles_opponent_port == 2 or singles_opponent_port == 3, \
      f"Invalid singles_opponent_port: {singles_opponent_port}. Must be 2 or 3."

  # Debug logging for singles mode
  '''if len(ports) == 2:
    print(f"get_game for singles mode - ports: {ports}, singles_opponent_port: {singles_opponent_port}")
    print(f"Players in game: {list(game.players.keys())}")'''

  players = {}
  for i, p in enumerate(ports):
    if p in game.players:
      players[f'p{i}'] = get_player(game.players[p])
    else:
      state = melee.PlayerState()
      state.action = melee.Action.DEAD_DOWN
      player = get_player(state)
      players[f'p{i}'] = player._replace(is_dead=True)

  if len(game.players) == 0:
    print("================== NO PLAYERS LEFT IN GAME ====================")

  # For singles mode, create a proper 4-player structure
  if len(ports) == 2:
    #print(f"Creating dummy players for singles mode - current players: {list(players.keys())}")
    
    # Create a dead player for empty slots
    state = melee.PlayerState()
    state.action = melee.Action.DEAD_DOWN
    state.position = melee.Position(100, 100)
    empty_player = get_player(state)._replace(is_dead=True)
    
    # Save the original players
    p0 = players['p0']
    p1 = players['p1']
    
    # Clear and rebuild the players dictionary with the correct mapping
    players = {
        'p0': p0,                                # Self (port 1)
        'p1': empty_player,                      # Teammate (empty in singles)
        'p2': p1 if singles_opponent_port == 2 else empty_player,  # Opponent 1
        'p3': p1 if singles_opponent_port == 3 else empty_player,  # Opponent 2
    }
    
    #print(f"Final players after singles mode processing: {list(players.keys())}")

  return Game(
      stage=game.stage.value,
      randall_phase=game.frame % 1200,
      is_teams=game.is_teams,
      **players,
  )

def get_slp(path: str) -> pa.StructArray:
  """Processes a slippi replay file."""
  console = melee.Console(is_dolphin=False,
                          allow_old_version=True,
                          path=path)
  console.connect()

  gamestate = console.step()
  ports = sorted(gamestate.players)
  if len(ports) != 2:
    raise InvalidGameError(f'Not a 2-player game.')

  frames = []

  while gamestate:
    if sorted(gamestate.player) != ports:
      raise InvalidGameError(f'Ports changed on frame {len(frames)}')
    game = get_game(gamestate)
    frames.append(nt_to_nest(game))
    gamestate = console.step()

  return pa.array(frames, type=GAME_TYPE)
