from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

import melee

from slippi_ai import utils
from slippi_ai.types import (
  GAME_TYPE,
  LIBMELEE_BUTTONS,
  Buttons,
  Controller,
  Game,
  InvalidGameError,
  Item,
  Items,
  Nana,
  Player,
  Randall,
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

_EMPTY_NANA = utils.map_nt(
    lambda t: t(0),
    utils.reify_tuple_type(Nana),
)

_EMPTY_ITEM = utils.map_nt(
    lambda t: t(0),
    utils.reify_tuple_type(Item),
)

def get_player(player: melee.PlayerState) -> Player:
  base = dict(
      percent=np.uint16(player.percent),
      facing=np.bool_(player.facing),
      x=np.float32(player.position.x),
      y=np.float32(player.position.y),
      action=np.uint16(player.action.value),
      invulnerable=np.bool_(player.invulnerable),
      character=np.uint8(player.character.value),
      jumps_left=np.uint8(player.jumps_left),
      shield_strength=np.float32(player.shield_strength),
      on_ground=np.bool_(player.on_ground),
      is_dead=np.bool_(player.stock == 0),
      stocks_left=np.uint8(player.stock),
      controller=get_controller(player.controller_state),
  )

  if player.nana is not None:
    nana_state = player.nana
    nana = Nana(
        exists=np.bool_(True),
        percent=np.uint16(nana_state.percent),
        facing=np.bool_(nana_state.facing),
        x=np.float32(nana_state.position.x),
        y=np.float32(nana_state.position.y),
        action=np.uint16(nana_state.action.value),
        invulnerable=np.bool_(nana_state.invulnerable),
        character=np.uint8(nana_state.character.value),
        jumps_left=np.uint8(nana_state.jumps_left),
        shield_strength=np.float32(nana_state.shield_strength),
        on_ground=np.bool_(nana_state.on_ground),
    )
  else:
    nana = _EMPTY_NANA

  return Player(
      nana=nana,
      **base,
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
  is_singles = len(ports) == 2

  if is_singles:
    #print(f"Creating dummy players for singles mode - current players: {list(players.keys())}")
    
    # Create a dead player for empty slots
    state = melee.PlayerState()
    state.action = melee.Action.DEAD_DOWN
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
      stage=np.uint8(game.stage.value),
      randall_phase=np.float32(game.frame % 1200),
      randall=Randall(
          x=np.float32(0.0),
          y=np.float32(0.0),
      ),
      items=Items(**{f'item_{i}': _EMPTY_ITEM for i in range(len(Items._fields))}),
      is_teams=not is_singles,
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
