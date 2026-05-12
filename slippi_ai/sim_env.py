import collections
import contextlib
import dataclasses
import typing as tp

import melee
import melee_sim
import numpy as np

from slippi_ai import dolphin
from slippi_ai import utils
from slippi_ai.envs import EnvOutput
from slippi_ai.types import Buttons, Controller, Game, Item, Items, Nana, Player, Randall, Stick


Port = int
Controllers = tp.Mapping[Port, Controller]

_SUPPORTED_PORTS = (1, 2)
_MELEE_TO_SIM_STAGE = {
    melee.Stage.FOUNTAIN_OF_DREAMS: melee_sim.Stage.FOUNTAIN_OF_DREAMS,
    melee.Stage.POKEMON_STADIUM: melee_sim.Stage.POKEMON_STADIUM,
    melee.Stage.YOSHIS_STORY: melee_sim.Stage.YOSHIS_STORY,
    melee.Stage.DREAMLAND: melee_sim.Stage.DREAM_LAND_N64,
    melee.Stage.BATTLEFIELD: melee_sim.Stage.BATTLEFIELD,
    melee.Stage.FINAL_DESTINATION: melee_sim.Stage.FINAL_DESTINATION,
}
_SIM_TO_MELEE_STAGE = {
    int(melee_sim.Stage.FOUNTAIN_OF_DREAMS): melee.Stage.FOUNTAIN_OF_DREAMS.value,
    int(melee_sim.Stage.POKEMON_STADIUM): melee.Stage.POKEMON_STADIUM.value,
    int(melee_sim.Stage.YOSHIS_STORY): melee.Stage.YOSHIS_STORY.value,
    int(melee_sim.Stage.DREAM_LAND_N64): melee.Stage.DREAMLAND.value,
    int(melee_sim.Stage.BATTLEFIELD): melee.Stage.BATTLEFIELD.value,
    int(melee_sim.Stage.FINAL_DESTINATION): melee.Stage.FINAL_DESTINATION.value,
}
SUPPORTED_STAGES = tuple(_MELEE_TO_SIM_STAGE.keys())

_TERMINAL_DTYPE = np.dtype(
    [
        ('frame_id', '<i4'),
        ('stage_id', '<u4'),
        ('done', 'u1'),
        ('match_ended', 'u1'),
        ('stockout', 'u1'),
        ('max_frame_reached', 'u1'),
        ('alive_count', 'u1'),
        ('alive_team_count', 'u1'),
        ('team_alive_mask', 'u1'),
        ('_pad0', 'u1'),
    ],
    align=False,
)


@dataclasses.dataclass(frozen=True)
class SimStepInfo:
  terminal: np.ndarray
  step_t: int


class SimBatchedEnvironment:
  """Batched melee_sim-backed environment with the existing slippi-ai env shape."""

  def __init__(
      self,
      num_envs: int,
      players: tp.Mapping[int, dolphin.Player] | None = None,
      *,
      length: int = 128,
      stage: melee.Stage | tp.Sequence[melee.Stage] = melee.Stage.FINAL_DESTINATION,
      max_frame_id: int = -1,
      data_dir: str | None = None,
      include_controller_state: bool = True,
  ):
    self._num_envs = int(num_envs)
    self._length = int(length)
    self._players = players or {
        1: dolphin.AI(melee.Character.FOX),
        2: dolphin.AI(melee.Character.FALCO),
    }
    self._ports = tuple(sorted(self._players))
    if self._ports != _SUPPORTED_PORTS:
      raise ValueError('SimBatchedEnvironment currently supports ports 1 and 2.')
    self._stage_by_env = _normalize_stages(stage, self._num_envs)
    self._max_frame_id = int(max_frame_id)
    self._include_controller_state = bool(include_controller_state)
    self.num_steps = 1

    self._env = melee_sim.EnvBatch(
        batch_size=self._num_envs,
        length=self._length,
        num_players=2,
        data_dir=data_dir,
    )
    self._buffers = self._env.buffers(action_format='controller')
    self._configure_all_matches()
    self._env.bind(self._buffers)
    self._env.reset_all()

    self._last_controllers = {
        port: neutral_controllers(self._num_envs) for port in self._ports
    }
    self._pending_reset = np.zeros(self._num_envs, dtype=np.bool_)
    self._last_step_info = SimStepInfo(
        terminal=np.zeros(self._num_envs, dtype=_TERMINAL_DTYPE),
        step_t=-1,
    )
    self._output_queue = collections.deque([
        self.current_state(needs_reset=np.ones(self._num_envs, dtype=np.bool_))
    ])

  def stop(self):
    self._env.close()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def current_state(self, needs_reset: np.ndarray | None = None) -> EnvOutput:
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_) if needs_reset is None else needs_reset
    frame = self._buffers.gamestate_view[self._env.t]
    return EnvOutput(
        gamestates={port: self._game_for_port(frame, port) for port in self._ports},
        needs_reset=np.asarray(needs_reset, dtype=np.bool_),
    )

  def reset(self, env_ids: tp.Sequence[int] | np.ndarray | None = None) -> EnvOutput:
    ids = np.arange(self._num_envs, dtype=np.int64) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= self._num_envs):
      raise ValueError('env_ids contains an out-of-range env index')
    self._ensure_cursor_room()
    reset_mask = self._buffers.reset_mask
    reset_mask[self._env.t, :] = 0
    reset_mask[self._env.t, ids] = 1
    self._env.reset_masked()
    reset_mask[self._env.t, ids] = 0
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_)
    needs_reset[ids] = True
    return self.current_state(needs_reset=needs_reset)

  def push(self, controllers: Controllers):
    self._output_queue.append(self._advance(controllers))

  def pop(self) -> EnvOutput:
    return self._output_queue.popleft()

  def peek(self) -> EnvOutput:
    return self._output_queue[0]

  def step(self, controllers: Controllers) -> EnvOutput:
    return self._advance(controllers)

  def multi_step(self, controllers: list[Controllers]) -> list[EnvOutput]:
    return [self.step(c) for c in controllers]

  @property
  def buffers(self):
    return self._buffers

  @property
  def cursor(self) -> int:
    return self._env.t

  @property
  def stages(self) -> np.ndarray:
    return self._stage_by_env.copy()

  @property
  def last_step_info(self) -> SimStepInfo:
    return self._last_step_info

  def _advance(self, controllers: Controllers) -> EnvOutput:
    self._ensure_cursor_room()
    if np.any(self._pending_reset):
      self.reset(np.flatnonzero(self._pending_reset))
      self._pending_reset[:] = False

    action = self._buffers.controller_action_view[self._env.t]
    for player_index, port in enumerate(self._ports):
      controller = controllers[port]
      _write_controller_action(action, controller, player_index)
      self._last_controllers[port] = controller

    step_t = self._env.t
    self._env.step(max_frame_id=self._max_frame_id)
    needs_reset = self._buffers.done[step_t].astype(np.bool_, copy=True)
    self._pending_reset[:] = needs_reset
    self._last_step_info = SimStepInfo(
        terminal=terminal_view(self._buffers)[step_t].copy(),
        step_t=step_t,
    )
    return self.current_state(needs_reset=needs_reset)

  def _ensure_cursor_room(self):
    if self._env.t >= self._length:
      self._env.reset_cursor()

  def _configure_all_matches(self):
    players = [
        melee_sim.PlayerConfig(character=_character_id(self._players[1])),
        melee_sim.PlayerConfig(character=_character_id(self._players[2])),
    ]
    self._env.configure_matches(
        self._buffers,
        [
            melee_sim.MatchConfig(
                stage=_MELEE_TO_SIM_STAGE[stage],
                players=tuple(players),
            )
            for stage in self._stage_by_env
        ],
    )

  def _game_for_port(self, frame: np.ndarray, port: Port) -> Game:
    slots_by_source = _slots_by_source(frame['slots'])
    self_source = port - 1
    opponent_source = 1 - self_source
    self_controller = self._last_controllers[port] if self._include_controller_state else neutral_controllers(self._num_envs)
    opponent_port = 1 if port == 2 else 2
    opponent_controller = (
        self._last_controllers[opponent_port]
        if self._include_controller_state
        else neutral_controllers(self._num_envs)
    )
    empty = _empty_player(self._num_envs)
    return Game(
        p0=_player_from_slot(slots_by_source[self_source], self_controller),
        p1=empty,
        p2=_player_from_slot(slots_by_source[opponent_source], opponent_controller),
        p3=empty,
        stage=_stage_array(frame['stage_id']),
        randall_phase=np.mod(frame['frame_id'], 1200).astype(np.float32),
        randall=Randall(
            x=frame['stage']['randall']['x'].astype(np.float32, copy=True),
            y=frame['stage']['randall']['y'].astype(np.float32, copy=True),
        ),
        items=_items_from_frame(frame['items']),
        is_teams=frame['is_teams'].astype(np.bool_, copy=True),
    )


def neutral_controllers(batch_size: int) -> Controller:
  shape = (int(batch_size),)
  return Controller(
      main_stick=Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      c_stick=Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      shoulder=np.zeros(shape, dtype=np.float32),
      buttons=Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in Buttons._fields
      }),
  )


def terminal_view(buffers: melee_sim.Buffers) -> np.ndarray:
  raw = buffers.terminal
  if raw.shape[2] < _TERMINAL_DTYPE.itemsize:
    raise ValueError('terminal buffer row is smaller than MslTerminal')
  return raw[:, :, :_TERMINAL_DTYPE.itemsize].view(_TERMINAL_DTYPE).reshape(raw.shape[0], raw.shape[1])


def supported_stages() -> tuple[melee.Stage, ...]:
  return SUPPORTED_STAGES


def _write_controller_action(action_frame: np.ndarray, controller: Controller, player_index: int):
  player = action_frame['p'][:, int(player_index)]
  player['main_stick_x'][:] = controller.main_stick.x
  player['main_stick_y'][:] = controller.main_stick.y
  player['c_stick_x'][:] = controller.c_stick.x
  player['c_stick_y'][:] = controller.c_stick.y
  player['shoulder'][:] = controller.shoulder
  for name in Buttons._fields:
    player['buttons'][name][:] = getattr(controller.buttons, name)


def _character_id(player: dolphin.Player) -> int:
  character = getattr(player, 'character', melee.Character.FOX)
  return int(character.value if isinstance(character, melee.Character) else character)


def _normalize_stages(stage: melee.Stage | tp.Sequence[melee.Stage], num_envs: int) -> np.ndarray:
  if isinstance(stage, melee.Stage):
    stages = [stage] * num_envs
  else:
    stages = list(stage)
    if len(stages) != num_envs:
      raise ValueError(f'stage sequence must have length num_envs={num_envs}')
  for item in stages:
    if item not in _MELEE_TO_SIM_STAGE:
      raise ValueError(f'SimBatchedEnvironment currently supports {SUPPORTED_STAGES}.')
  return np.asarray(stages, dtype=object)


def _slots_by_source(slots: np.ndarray) -> dict[int, np.ndarray]:
  result = {}
  for i in range(slots.shape[1]):
    if not np.any(slots[:, i]['present']):
      continue
    source = slots[:, i]['source_player']
    if np.all(source == source[0]):
      result[int(source[0])] = slots[:, i]
  if 0 not in result or 1 not in result:
    raise RuntimeError('melee_sim gamestate did not contain source players 0 and 1')
  return result


def _player_from_slot(slot: np.ndarray, controller: Controller) -> Player:
  present = slot['present'].astype(np.bool_, copy=False)
  stocks = slot['stocks'].astype(np.uint8, copy=True)
  return Player(
      percent=np.rint(slot['percent']).clip(0, np.iinfo(np.uint16).max).astype(np.uint16),
      facing=slot['facing'].astype(np.bool_, copy=True),
      x=slot['pos_x'].astype(np.float32, copy=True),
      y=slot['pos_y'].astype(np.float32, copy=True),
      action=slot['action_id'].astype(np.uint16, copy=True),
      invulnerable=slot['invulnerable'].astype(np.bool_, copy=True),
      character=slot['char_id'].astype(np.uint8, copy=True),
      jumps_left=slot['jumps_left'].astype(np.uint8, copy=True),
      shield_strength=slot['shield_hp'].astype(np.float32, copy=True),
      on_ground=slot['on_ground'].astype(np.bool_, copy=True),
      is_dead=np.logical_or(np.logical_not(present), stocks == 0),
      stocks_left=stocks,
      controller=controller,
      nana=_empty_nana(slot.shape[0]),
  )


def _empty_player(batch_size: int) -> Player:
  batch_size = int(batch_size)
  zeros_bool = np.zeros(batch_size, dtype=np.bool_)
  zeros_f32 = np.zeros(batch_size, dtype=np.float32)
  zeros_u16 = np.zeros(batch_size, dtype=np.uint16)
  zeros_u8 = np.zeros(batch_size, dtype=np.uint8)
  return Player(
      percent=zeros_u16.copy(),
      facing=zeros_bool.copy(),
      x=zeros_f32.copy(),
      y=zeros_f32.copy(),
      action=zeros_u16.copy(),
      invulnerable=zeros_bool.copy(),
      character=zeros_u8.copy(),
      jumps_left=zeros_u8.copy(),
      shield_strength=zeros_f32.copy(),
      on_ground=zeros_bool.copy(),
      is_dead=np.ones(batch_size, dtype=np.bool_),
      stocks_left=zeros_u8.copy(),
      controller=neutral_controllers(batch_size),
      nana=_empty_nana(batch_size),
  )


def _empty_nana(batch_size: int) -> Nana:
  zeros_bool = np.zeros(batch_size, dtype=np.bool_)
  zeros_f32 = np.zeros(batch_size, dtype=np.float32)
  zeros_u16 = np.zeros(batch_size, dtype=np.uint16)
  zeros_u8 = np.zeros(batch_size, dtype=np.uint8)
  return Nana(
      exists=zeros_bool.copy(),
      percent=zeros_u16.copy(),
      facing=zeros_bool.copy(),
      x=zeros_f32.copy(),
      y=zeros_f32.copy(),
      action=zeros_u16.copy(),
      invulnerable=zeros_bool.copy(),
      character=zeros_u8.copy(),
      jumps_left=zeros_u8.copy(),
      shield_strength=zeros_f32.copy(),
      on_ground=zeros_bool.copy(),
  )


def _items_from_frame(items: np.ndarray) -> Items:
  return Items(**{
      f'item_{i}': Item(
          exists=items[:, i]['exists'].astype(np.bool_, copy=True),
          type=items[:, i]['type'].astype(np.uint16, copy=True),
          state=items[:, i]['state'].astype(np.uint8, copy=True),
          x=items[:, i]['pos_x'].astype(np.float32, copy=True),
          y=items[:, i]['pos_y'].astype(np.float32, copy=True),
      )
      for i in range(len(Items._fields))
  })


def _stage_array(stage_id: np.ndarray) -> np.ndarray:
  out = np.zeros(stage_id.shape, dtype=np.uint8)
  for sim_stage, melee_stage in _SIM_TO_MELEE_STAGE.items():
    out[stage_id == sim_stage] = melee_stage
  return out
