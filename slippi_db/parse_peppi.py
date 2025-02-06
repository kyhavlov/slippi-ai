import numpy as np
import pyarrow as pa

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

def get_buttons(button_bits: np.ndarray) -> types.Buttons:
  return types.Buttons(**{
      name: np.asarray(
          np.bitwise_and(button_bits, BUTTON_MASKS[button]),
          dtype=bool)
      for name, button in types.LIBMELEE_BUTTONS.items()
  })

def to_libmelee_stick(raw_stick: np.ndarray) -> np.ndarray:
  return (raw_stick / 2.) + 0.5

def get_stick(stick) -> types.Stick:
  return types.Stick(
      x=to_libmelee_stick(np.nan_to_num(stick.field('x').to_numpy(zero_copy_only=False), nan=0.)),
      y=to_libmelee_stick(np.nan_to_num(stick.field('y').to_numpy(zero_copy_only=False), nan=0.)),
  )

def get_player(player: pa.StructArray) -> types.Player:
  leader = player.field('leader')

  post = leader.field('post')
  get_post = lambda key: post.field(key)
  position = post.field('position')
  pre = leader.field('pre')

  dead = np.where(np.isnan(position.field('x')), True, False)
  char = get_post('character')[0].as_py()

  player = types.Player(
      percent=np.asarray(np.nan_to_num(get_post('percent'), nan=np.uint16(0)), dtype=np.uint16),
      facing=get_post('direction').to_numpy(zero_copy_only=False) > 0,
      x=np.nan_to_num(position.field('x'), nan=0.),
      y=np.nan_to_num(position.field('y'), nan=0.),
      action=np.nan_to_num(get_post('state'), nan=np.uint16(0)),
      # libmelee does extra processing to determine invulnerability
      invulnerable=get_post('hurtbox_state').to_numpy(zero_copy_only=False) != 0,
      character=np.nan_to_num(get_post('character'), nan=char),  # uint8
      jumps_left=np.nan_to_num(get_post('jumps'), nan=np.uint8(0)),  # uint8
      shield_strength=np.nan_to_num(get_post('shield'), nan=0.),  # float
      controller=types.Controller(
          main_stick=get_stick(pre.field('joystick')),
          c_stick=get_stick(pre.field('cstick')),
          # libmelee reads the logical value and assigns it to both l/r
          shoulder=np.nan_to_num(pre.field('triggers'), nan=0.),
          buttons=get_buttons(pre.field('buttons_physical').fill_null(0)),
      ),
      on_ground=np.logical_not(
          post.field('airborne').to_numpy(zero_copy_only=False)),
      is_dead=dead,
  )

  return player

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

  players = {}
  port_names = sorted(p['port'] for p in game.start['players'])
  ports_data = frames.field('ports')
  #print(game.metadata)
  #print(game.start)
  #print(game.frames[0])
  for i, port_name in enumerate(port_names):
    players[f'p{i}'] = get_player(ports_data.field(port_name))
    #player: types.Player = players[f'p{i}']
    '''fields = [player.percent, player.facing, player.x, player.y, player.action, 
                player.invulnerable, player.character, player.jumps_left, 
                player.shield_strength, player.on_ground, player.controller.main_stick.x,
                player.controller.main_stick.y, player.controller.c_stick.x, player.controller.c_stick.y,
                player.controller.shoulder, player.controller.buttons.A, player.controller.buttons.B,
                player.controller.buttons.X, player.controller.buttons.Y, player.controller.buttons.Z,
                player.controller.buttons.L, player.controller.buttons.R, player.controller.buttons.D_UP]

    was_nan = False
    for j, field in enumerate(fields):
      nan_present = np.any(np.isnan(field))
      if nan_present:
        nan_locs = np.argwhere(np.isnan(field)).flatten()
        action = player.action[nan_locs[0]]
        print(f'NAN in field {j} for player {i} (action: {action}) in {game.metadata}: ', nan_locs)
        was_nan = True

    assert not was_nan'''

  if len(game.start['players']) == 2:
    players['p2'] = zero_out_namedtuple(players['p0'])
    players['p2'].is_dead.fill(True)
    players['p3'] = zero_out_namedtuple(players['p0'])
    players['p3'].is_dead.fill(True)

    '''print(players['p0'])
    print(len(players['p0'].x))
    print(len(players['p0'].shield_strength))
    print('=========================================')
    print('=========================================')
    print('=========================================')
    print(players['p1'])
    print(len(players['p1'].x))
    print(len(players['p1'].shield_strength))
    print('=========================================')
    print('=========================================')
    print('=========================================')
    print(players['p2'])
    print(len(players['p2'].x))
    print(len(players['p2'].shield_strength))
    print('=========================================')
    print('=========================================')
    print('=========================================')
    print(players['p3'])
    print(len(players['p3'].x))
    print(len(players['p3'].shield_strength))'''

  stage = melee.enums.to_internal_stage(game.start['stage'])
  stage = np.full([len(frames)], stage.value, dtype=np.uint8)

  game = types.Game(stage=stage, **players)
  game_array = types.array_from_nt(game)

  index = frames.field('id').to_numpy()
  first_indices = []
  next_idx = -123
  for i, idx in enumerate(index):
    if idx == next_idx:
      first_indices.append(i)
      next_idx += 1
  return game_array.take(first_indices)

def get_slp(path: str) -> types.GAME_TYPE:
  game = peppi_py.read_slippi(path)
  return from_peppi(game)
