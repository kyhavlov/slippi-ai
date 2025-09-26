import numpy as np

import melee
from melee import Button
import peppi_py

from slippi_ai import types

BUTTON_MASKS = {
    Button.BUTTON_A: 0x0100,
    Button.BUTTON_B: 0x0200,
    Button.BUTTON_X: 0x0400,
    Button.BUTTON_Y: 0x0800,
    Button.BUTTON_START: 0x1000,
    Button.BUTTON_Z: 0x0010,
    Button.BUTTON_R: 0x0020,
    Button.BUTTON_L: 0x0040,
    Button.BUTTON_D_LEFT: 0x0001,
    Button.BUTTON_D_RIGHT: 0x0002,
    Button.BUTTON_D_DOWN: 0x0004,
    Button.BUTTON_D_UP: 0x0008,
}


def _arrow_to_numpy(array, *, dtype=None, nan=0.0):
  """Convert a pyarrow array into numpy while handling NaNs."""

  np_array = array.to_numpy(zero_copy_only=False)
  if np.issubdtype(np_array.dtype, np.floating):
    np_array = np.nan_to_num(np_array, nan=nan)

  if dtype is not None:
    np_array = np_array.astype(dtype, copy=False)

  return np_array


def _to_bool(array) -> np.ndarray:
  return array.to_numpy(zero_copy_only=False).astype(bool, copy=False)


def _to_float(array) -> np.ndarray:
  return np.nan_to_num(
      array.to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
      nan=0.0)


def _to_uint(array, dtype) -> np.ndarray:
  arr = np.nan_to_num(array.to_numpy(zero_copy_only=False), nan=0.0)
  return arr.astype(dtype, copy=False)


def _libmelee_trigger(array) -> np.ndarray:
  return np.nan_to_num(
      array.to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
      nan=0.0)


def _libmelee_stick(position) -> types.Stick:
  x = _to_float(position.x)
  y = _to_float(position.y)
  return types.Stick(
      x=(x / 2.0) + 0.5,
      y=(y / 2.0) + 0.5,
  )


def _controller_from_pre(pre) -> types.Controller:
  button_bits = np.nan_to_num(
      pre.buttons_physical.to_numpy(zero_copy_only=False),
      nan=0.0,
  ).astype(np.uint32, copy=False)
  buttons = types.Buttons(**{
      name: np.asarray(np.bitwise_and(button_bits, BUTTON_MASKS[button]), dtype=bool)
      for name, button in types.LIBMELEE_BUTTONS.items()
  })

  return types.Controller(
      main_stick=_libmelee_stick(pre.joystick),
      c_stick=_libmelee_stick(pre.cstick),
      shoulder=_libmelee_trigger(pre.triggers),
      buttons=buttons,
  )


def _player_from_port(port: peppi_py.frame.PortData) -> types.Player:
  leader = port.leader
  pre = leader.pre
  post = leader.post

  position_x_raw = post.position.x.to_numpy(zero_copy_only=False)
  position_y_raw = post.position.y.to_numpy(zero_copy_only=False)
  direction_raw = post.direction.to_numpy(zero_copy_only=False)
  percent_raw = post.percent.to_numpy(zero_copy_only=False)
  state_raw = post.state.to_numpy(zero_copy_only=False)
  shield_raw = post.shield.to_numpy(zero_copy_only=False)
  jumps_raw = post.jumps.to_numpy(zero_copy_only=False)
  character_raw = post.character.to_numpy(zero_copy_only=False)
  stocks_raw = post.stocks.to_numpy(zero_copy_only=False)
  airborne_raw = post.airborne.to_numpy(zero_copy_only=False)

  position_x = np.nan_to_num(position_x_raw, nan=0.0).astype(np.float32, copy=False)
  position_y = np.nan_to_num(position_y_raw, nan=0.0).astype(np.float32, copy=False)
  direction = np.nan_to_num(direction_raw, nan=0.0)
  percent = np.nan_to_num(percent_raw, nan=0.0)
  state = np.nan_to_num(state_raw, nan=0.0)
  shield = np.nan_to_num(shield_raw, nan=0.0).astype(np.float32, copy=False)
  jumps = np.nan_to_num(jumps_raw, nan=0.0)
  character = np.nan_to_num(character_raw, nan=0.0)
  stocks = np.nan_to_num(stocks_raw, nan=0.0)
  airborne = np.nan_to_num(airborne_raw, nan=0.0)

  return types.Player(
      percent=percent.astype(np.uint16, copy=False),
      facing=direction.astype(np.float32, copy=False) > 0,
      x=position_x,
      y=position_y,
      action=state.astype(np.uint16, copy=False),
      invulnerable=((post.hurtbox_state.to_numpy(zero_copy_only=False)
                     if post.hurtbox_state is not None else np.zeros_like(position_x_raw)) != 0),
      character=character.astype(np.uint8, copy=False),
      jumps_left=jumps.astype(np.uint8, copy=False),
      shield_strength=shield,
      on_ground=np.logical_not(airborne.astype(bool, copy=False)),
      is_dead=np.isnan(position_x_raw),
      stocks_left=stocks.astype(np.uint8, copy=False),
      controller=_controller_from_pre(pre),
  )

# Create a copy of the given player but with all fields zeroed out
def zero_out_namedtuple(nt: types.NamedTuple) -> types.NamedTuple:
  def zero_out_field(field):
    if isinstance(field, np.ndarray):
      return np.zeros_like(field)  # Maintain shape and type
    elif isinstance(field, (np.uint16, np.bool_, np.float32, np.uint8)):
      return type(field)(0)  # Return zero of the same type
    elif isinstance(field, tuple) and hasattr(field, "_fields"):  # Check for NamedTuple
      return zero_out_namedtuple(field)  # Recursively zero out subfields
    else:
      return field  # Return as is for non-handled types

  zeroed_fields = {key: zero_out_field(getattr(nt, key)) for key in nt._fields}
  return type(nt)(**zeroed_fields)


def from_peppi(game: peppi_py.Game) -> types.GAME_TYPE:
  frames = game.frames

  port_data = list(frames.ports)
  players_map: dict[str, types.Player] = {}

  for idx, port in enumerate(port_data):
    players_map[f'p{idx}'] = _player_from_port(port)

  if len(port_data) == 2:
    p0 = players_map['p0']
    p1 = players_map['p1']
    empty = zero_out_namedtuple(p0)
    empty.is_dead.fill(True)

    players_map = {
        'p0': p0,
        'p1': empty,
        'p2': p1,
        'p3': zero_out_namedtuple(p0),
    }
    players_map['p3'].is_dead.fill(True)
  elif len(port_data) != 4:
    raise ValueError(f'Unexpected port count: {len(port_data)}')

  frame_ids = frames.id.to_numpy(zero_copy_only=False)
  game_length = len(frame_ids)

  stage = melee.enums.to_internal_stage(game.start.stage)
  stage_array = np.full(game_length, stage.value, dtype=np.uint8)
  is_teams = np.full(game_length, len(port_data) == 4, dtype=np.bool_)
  randall_phase = (np.arange(game_length, dtype=np.float32) % 1200).astype(np.float32)

  game_nt = types.Game(
      stage=stage_array,
      randall_phase=randall_phase,
      is_teams=is_teams,
      **players_map,
  )
  return types.array_from_nt(game_nt)

def get_slp(path: str) -> types.GAME_TYPE:
  game = peppi_py.read_slippi(path)
  return from_peppi(game)
