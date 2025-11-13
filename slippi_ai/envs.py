import collections
import contextlib
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
import typing as tp
from typing import Mapping, Optional
import time

import numpy as np
import portpicker

from melee.slippstream import EnetDisconnected
from melee import GameState, Stage, enums

from slippi_ai import dolphin, utils
from slippi_ai.controller_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_ai import data
from slippi_db.parse_libmelee import get_game
from slippi_ai import match_reporting
import signal

Port = int
Controllers = Mapping[Port, Controller]

DOUBLES_PORT_MAPPINGS: Mapping[int, list[int]] = {
    1: (1, 4, 2, 3),
    2: (2, 3, 1, 4),
    3: (3, 2, 4, 1),
    4: (4, 1, 3, 2),
}

SINGLES_PORT_MAPPINGS: Mapping[int, list[int]] = {
    1: (1, 2),
    2: (2, 1),
    3: (1, 2),
    4: (2, 1),
}


class MixedModeLayout(tp.NamedTuple):
  total_envs: int
  inner_batch_size: int
  num_single_groups: int
  num_double_groups: int
  num_single_chunks: int
  num_double_chunks: int
  chunk_modes: list[str]
  chunk_physical_sizes: list[int]
  singles_multiplier: int


def compute_mixed_mode_layout(
    *,
    total_envs: int,
    singles_ratio: float,
    inner_batch_size: int,
    min_single_envs: int = 0,
    singles_multiplier: int = 2,
) -> MixedModeLayout:
  if not 0 <= singles_ratio <= 1:
    raise ValueError('singles_ratio must be between 0 and 1.')
  if total_envs <= 0:
    raise ValueError('total_envs must be positive.')
  if inner_batch_size <= 0:
    raise ValueError('inner_batch_size must be positive.')
  if singles_multiplier < 1:
    raise ValueError('singles_multiplier must be positive.')

  outer_batch_size, remainder = divmod(total_envs, inner_batch_size)
  if remainder:
    raise ValueError('total_envs must be divisible by inner_batch_size.')

  tentative_singles = int(round(total_envs * singles_ratio))
  tentative_singles = max(tentative_singles, min_single_envs)

  if tentative_singles == 0:
    logging.warning(
        'Mixed-mode layout collapsed to doubles-only (total_envs=%d, '
        'singles_ratio=%.3f, inner_batch_size=%d).',
        total_envs, singles_ratio, inner_batch_size)
    chunk_modes = ['doubles'] * outer_batch_size
    chunk_physical_sizes = [inner_batch_size] * outer_batch_size
    return MixedModeLayout(
        total_envs=total_envs,
        inner_batch_size=inner_batch_size,
        num_single_groups=0,
        num_double_groups=total_envs,
        num_single_chunks=0,
        num_double_chunks=outer_batch_size,
        chunk_modes=chunk_modes,
        chunk_physical_sizes=chunk_physical_sizes,
        singles_multiplier=singles_multiplier,
    )

  if inner_batch_size % 2 != 0:
    raise ValueError('inner_batch_size must be even when singles are enabled.')

  if tentative_singles % 2 != 0:
    tentative_singles += 1

  if tentative_singles > total_envs:
    tentative_singles = total_envs

  remainder = tentative_singles % inner_batch_size
  if remainder:
    rounded_up = tentative_singles + (inner_batch_size - remainder)
    if rounded_up <= total_envs:
      tentative_singles = rounded_up
    else:
      tentative_singles -= remainder

  if tentative_singles == 0:
    logging.warning(
        'Mixed-mode layout collapsed to doubles-only after rounding '
        '(total_envs=%d, singles_ratio=%.3f, inner_batch_size=%d).',
        total_envs, singles_ratio, inner_batch_size)
    chunk_modes = ['doubles'] * outer_batch_size
    chunk_physical_sizes = [inner_batch_size] * outer_batch_size
    return MixedModeLayout(
        total_envs=total_envs,
        inner_batch_size=inner_batch_size,
        num_single_groups=0,
        num_double_groups=total_envs,
        num_single_chunks=0,
        num_double_chunks=outer_batch_size,
        chunk_modes=chunk_modes,
        chunk_physical_sizes=chunk_physical_sizes,
        singles_multiplier=singles_multiplier,
    )

  num_single_chunks = tentative_singles // inner_batch_size
  num_single_groups = num_single_chunks * inner_batch_size
  num_double_groups = total_envs - num_single_groups
  num_double_chunks = outer_batch_size - num_single_chunks

  chunk_modes = (
      ['singles'] * num_single_chunks +
      ['doubles'] * num_double_chunks
  )

  chunk_physical_sizes = [
      inner_batch_size * singles_multiplier
      if mode == 'singles' else inner_batch_size
      for mode in chunk_modes
  ]

  if num_double_groups == 0:
    logging.warning(
        'Mixed-mode layout rounded to singles-only groups '
        '(total_envs=%d, singles_ratio=%.3f, inner_batch_size=%d).',
        total_envs, singles_ratio, inner_batch_size)
  elif num_single_groups == 0:
    logging.warning(
        'Mixed-mode layout rounded to doubles-only groups '
        '(total_envs=%d, singles_ratio=%.3f, inner_batch_size=%d).',
        total_envs, singles_ratio, inner_batch_size)

  return MixedModeLayout(
      total_envs=total_envs,
      inner_batch_size=inner_batch_size,
      num_single_groups=num_single_groups,
      num_double_groups=num_double_groups,
      num_single_chunks=num_single_chunks,
      num_double_chunks=num_double_chunks,
      chunk_modes=chunk_modes,
      chunk_physical_sizes=chunk_physical_sizes,
      singles_multiplier=singles_multiplier,
  )


def compose_singles_group(
    left_state: GameState,
    right_state: GameState,
) -> Mapping[int, Game]:
  gamestates: dict[int, Game] = {}
  for port in (1, 2):
    gamestates[port] = get_game(
        left_state,
        ports=SINGLES_PORT_MAPPINGS[port],
        singles_opponent_port=2,
    )
  for port in (3, 4):
    gamestates[port] = get_game(
        right_state,
        ports=SINGLES_PORT_MAPPINGS[port],
        singles_opponent_port=3,
    )
  return gamestates

def is_initial_frame(gamestate: GameState) -> bool:
  return gamestate.frame == -123

class EnvOutput(tp.NamedTuple):
  gamestates: Mapping[int, Game]
  needs_reset: bool
  needs_reset_ports: Optional[Mapping[int, tp.Any]] = None


def _merge_singles_env_outputs(
    left_output: 'EnvOutput',
    right_output: 'EnvOutput',
) -> 'EnvOutput':
  gamestates = dict(left_output.gamestates)
  gamestates.update(right_output.gamestates)

  left_reset = left_output.needs_reset
  right_reset = right_output.needs_reset

  if isinstance(left_reset, np.ndarray) or isinstance(right_reset, np.ndarray):
    needs_reset = np.logical_or(np.asarray(left_reset), np.asarray(right_reset))
  else:
    needs_reset = bool(left_reset) or bool(right_reset)

  def _ports_from_output(output: 'EnvOutput', fallback: tp.Any) -> dict[int, tp.Any]:
    if output.needs_reset_ports is not None:
      return dict(output.needs_reset_ports)
    return {port: fallback for port in output.gamestates}

  needs_reset_ports: dict[int, tp.Any] = {}
  needs_reset_ports.update(_ports_from_output(left_output, left_reset))
  needs_reset_ports.update(_ports_from_output(right_output, right_reset))

  return EnvOutput(gamestates, needs_reset, needs_reset_ports)

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
  def handler(signum, frame):
    raise TimeoutError()

  # set the timeout handler
  signal.signal(signal.SIGALRM, handler) 
  signal.alarm(timeout_duration)
  try:
    result = func(*args, **kwargs)
  except TimeoutError as exc:
    result = default
  finally:
    signal.alarm(0)

  return result

class Environment:
  """Wraps dolphin to provide an RL interface."""

  def __init__(
      self,
      dolphin_kwargs: dict,
      agent_names: tuple[str, str] = [],
      swap_ports: bool = False,
      check_controller_outputs: bool = False,
      enable_singles: bool = False,
      singles_role: tp.Optional[str] = None,
  ):
    players: dict[Port, dolphin.Player] = dolphin_kwargs['players']
    if len(players) != 4:
      raise ValueError('Environment requires exactly 4 players.')
    
    self._agent_names = agent_names
    self._enable_singles = enable_singles
    self._singles_role = singles_role
    self._env_port = dolphin_kwargs.get('slippi_port')

    ports = list(players)
    actual_ports = list(reversed(ports)) if swap_ports else ports
    self.port_to_actual = dict(zip(ports, actual_ports))
    self.port_from_actual = dict(zip(actual_ports, ports))
    self._current_characters = []

    actual_players = {
        actual_port: players[port]
        for port, actual_port in self.port_to_actual.items()
    }

    if not enable_singles:
      self._dolphin_kwargs = dict(
          dolphin_kwargs,
          players=actual_players,
          desired_teams={1: 0, 2: 1, 3: 1, 4: 0},
      )
      self._dolphin = dolphin.Dolphin(**self._dolphin_kwargs)
      self._controlled_ports = (1, 2, 3, 4)
      self._port_map = self.port_to_actual
    else:
      if singles_role not in {'left', 'right'}:
        raise ValueError('singles_role must be "left" or "right" when enable_singles=True.')

      if singles_role == 'left':
        singles_players = {1: players[1], 2: players[2]}
        self._controlled_ports = (1, 2)
        self._port_map = {1: 1, 2: 2}
      else:
        singles_players = {1: players[3], 2: players[4]}
        self._controlled_ports = (3, 4)
        self._port_map = {3: 1, 4: 2}

      self._dolphin_kwargs = dict(dolphin_kwargs, players=singles_players)
      self._dolphin = dolphin.Dolphin(**self._dolphin_kwargs)

    self._dead_frame = {port: 0 for port in ports}

    self._prev_state: Optional[GameState] = None

  def stop(self):
    self._dolphin.stop()

  def start(self):
    self._dolphin = dolphin.Dolphin(**self._dolphin_kwargs)

  def current_state(self) -> EnvOutput:
    if self._prev_state is None:
      self._prev_state = self._dolphin.step()

    games = {}

    if not self._enable_singles:
      needs_reset = is_initial_frame(self._prev_state)
      for port, ports in DOUBLES_PORT_MAPPINGS.items():
        games[port] = get_game(self._prev_state, ports)
      needs_reset_ports = {port: needs_reset for port in games}
    else:
      needs_reset = is_initial_frame(self._prev_state)
      needs_reset_ports = {}
      for port in self._controlled_ports:
        ports = SINGLES_PORT_MAPPINGS[port]
        singles_opponent_port = 2 if port < 3 else 3
        games[port] = get_game(self._prev_state, ports, singles_opponent_port)
        needs_reset_ports[port] = needs_reset

    #print("current_state keys: ", self._enable_singles, games.keys())

    return EnvOutput(games, needs_reset, needs_reset_ports)

  def multi_current_state(self) -> list[EnvOutput]:
    return [self.current_state()]
  
  def _step(self, controllers: Controllers) -> EnvOutput:
    """Send controllers for each AI. Return the next state."""
    if not self._enable_singles:
      for port, controller in controllers.items():
        actual_port = self.port_to_actual[port]
        send_controller(self._dolphin.controllers[actual_port], controller)
    else:
      for port in self._controlled_ports:
        controller = controllers.get(port)
        if controller is None:
          continue
        actual_port = self._port_map[port]
        send_controller(self._dolphin.controllers[actual_port], controller)

    game_was_over = False if self._prev_state is None else match_reporting.match_is_over(self._prev_state)

    # TODO: compute reward?
    self._prev_state = self._dolphin.step()

    if self._prev_state.frame == 1:
      self._current_characters = [player.character for player in self._prev_state.players.values()]

    # stock stealing hack
    if not self._enable_singles:
      for port, controller in controllers.items():
        player_port = port
        teammate_port = DOUBLES_PORT_MAPPINGS[port][1]

        if player_port not in self._prev_state.players or self._prev_state.players[player_port].stock == 0:
          self._dead_frame[port] += 1
          if self._dead_frame[port] >= 120 and teammate_port in self._prev_state.players and self._prev_state.players[teammate_port].stock > 1:
            logging.info("port %d is dead, stock stealing from %d, %s", player_port, teammate_port, self._prev_state.players)
            logging.info("pressing start")
            self._dolphin.controllers[player_port].press_button(enums.Button.BUTTON_START)
        else:
          self._dolphin.controllers[player_port].release_button(enums.Button.BUTTON_START)
          self._dead_frame[port] = 0

    return self.current_state()

  def step(
    self,
    controllers: Controllers,
  ) -> EnvOutput:
    state = timeout(self._step, args=(controllers,), timeout_duration=10, default=None)

    # reset env if step timed out
    if state is None:
      frame = -200
      stage = "unknown"
      characters = []
      if self._prev_state is not None:
        frame = self._prev_state.frame
        stage = self._prev_state.stage
        characters = [player.character for player in self._prev_state.players.values()]
      raise TimeoutError("step timed out for 10s, frame: " + str(frame) + ", stage: " + str(stage) + ", characters: " + str(characters))

    return state

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

T = tp.TypeVar('T')


class SafeEnvironment:
  """Wraps an environment with retries on disconnect."""

  def __init__(
      self,
      dolphin_kwargs: dict,
      num_retries: int = 4,
      agent_names: tuple[str, str] = [],
      singles_role: tp.Optional[str] = None,
      **env_kwargs,
  ):
    self._dolphin_kwargs = dolphin_kwargs.copy()
    self._num_retries = num_retries
    self._env_kwargs = env_kwargs
    if singles_role is not None:
      self._env_kwargs = dict(self._env_kwargs, singles_role=singles_role)
    self._agent_names = agent_names
    self._build_environment()

  def _reset_port(self):
    old_port = self._dolphin_kwargs['slippi_port']
    new_port = portpicker.pick_unused_port()
    logging.warning('Switching from port %d to port %d.', old_port, new_port)
    self._dolphin_kwargs['slippi_port'] = new_port

  def _build_environment(self):
    self._env = utils.retry(
        lambda: Environment(self._dolphin_kwargs, agent_names=self._agent_names, **self._env_kwargs),
        on_exception={dolphin.ConnectFailed: self._reset_port},
        num_retries=2)

  def _reset_env(self):
    self._env.stop()  # closes associated dolphin instances, freeing up ports
    self._build_environment()

  # def _retry(self, method: str, *args):
  #   return retry(
  #       lambda: getattr(self._env, method)(*args),
  #       on_exception={dolphin.ConnectFailed: self._reset_env},
  #       num_retries=self._num_retries)

  def _retry(self, f: tp.Callable[[], T]) -> T:
    return utils.retry(
        f,
        on_exception={
            EnetDisconnected: self._reset_env,
            TimeoutError: self._reset_env,
        },
        num_retries=self._num_retries)

  def current_state(self) -> EnvOutput:
    return self._retry(lambda: self._env.current_state())

  def multi_current_state(self) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_current_state())

  def step(
    self,
    controllers: Controllers,
  ) -> EnvOutput:
    return self._retry(lambda: self._env.step(controllers))

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_step(controllers))

  def stop(self):
    self._env.stop()


class BatchedEnvironment:
  """A set of synchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dolphin_kwargs: dict,
      slippi_ports: Optional[list[int]] = None,
      num_retries: int = 2,
      agent_names: list[tuple[str, str]] = [],
      swap_ports: bool = True,  # Swap ports on half of the environments.
      enable_singles: bool = False,  # Enable singles mode for half the envs
      singles_roles: Optional[list[tp.Optional[str]]] = None,
  ):
    self._dolphin_kwargs = dolphin_kwargs
    slippi_ports = slippi_ports or utils.find_open_udp_ports(num_envs)

    if enable_singles:
      if singles_roles is None:
        raise ValueError('singles_roles must be provided when enable_singles=True.')
      if len(singles_roles) != num_envs:
        raise ValueError('singles_roles length must match num_envs.')
    else:
      if singles_roles is not None:
        raise ValueError('singles_roles is only valid when enable_singles=True.')

    if swap_ports and num_envs % 2 != 0:
      raise ValueError('swap_ports=True requires an even number of environments.')
    
    envs: list[SafeEnvironment] = []
    for i in range(num_envs):
      dolphin_kwargs_i = dolphin_kwargs.copy()
      dolphin_kwargs_i.update(slippi_port=slippi_ports[i])
      env = SafeEnvironment(
          dolphin_kwargs_i,
          num_retries=num_retries,
          agent_names=agent_names[i],
          swap_ports=swap_ports and i >= num_envs // 2,
          enable_singles=enable_singles,
          singles_role=singles_roles[i] if singles_roles else None,
      )
      envs.append(env)

    self._envs = envs

    # Optional "async" interface for compatibility with the Async* Envs.
    self._output_queue = collections.deque()
    self._output_queue.appendleft(self.current_state())

  @property
  def num_steps(self) -> int:
    return 1

  def stop(self):
    for env in self._envs:
      env.stop()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def current_state(self) -> EnvOutput:
    current_states = [env.current_state() for env in self._envs]
    return utils.batch_nest_nt(current_states)

  def multi_current_state(self) -> list[EnvOutput]:
    return [self.current_state()]

  def step(
    self,
    controllers: Controllers,
  ) -> EnvOutput:
    get_action = lambda i: utils.map_single_structure(
        lambda x: x[i], controllers)

    results = [
        env.step(get_action(i))
        for i, env in enumerate(self._envs)
    ]
    return utils.batch_nest_nt(results)

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

  def push(self, controllers: Controllers):
    self._output_queue.appendleft(self.step(controllers))

  def pop(self) -> EnvOutput:
    return self._output_queue.pop()

  def peek(self) -> EnvOutput:
    return self._output_queue[-1]

def build_environment(
    num_envs: int,  # zero means unbatched env
    dolphin_kwargs: dict,
    slippi_ports: Optional[list[int]] = None,
    num_retries: int = 2,
    agent_names: list[tuple[str, str]] = [],
    singles_roles: Optional[list[tp.Optional[str]]] = None,
    **env_kwargs,
) -> tp.Union[SafeEnvironment, BatchedEnvironment]:
  if num_envs == 0:
    if slippi_ports:
      assert len(slippi_ports) == 1
      dolphin_kwargs = dolphin_kwargs.copy()
      dolphin_kwargs['slippi_port'] = slippi_ports[0]
    singles_role = singles_roles[0] if singles_roles else None
    return SafeEnvironment(
        dolphin_kwargs,
        num_retries=num_retries,
        agent_names=agent_names[0],
        singles_role=singles_role,
        **env_kwargs,
    )

  # BatchedEnvironment uses SafeEnvironment internally
  return BatchedEnvironment(
      num_envs,
      dolphin_kwargs,
      slippi_ports,
      num_retries,
      agent_names,
      singles_roles=singles_roles,
      **env_kwargs,
  )

def _run_env(
    build_env_kwargs: dict,
    conn: Connection,
    # output_queue: mp.Queue,
    # stop: Event,
    batch_time: bool = False,
):
  send = conn.send
  # send = output_queue.put

  env = None
  try:
    env = build_environment(**build_env_kwargs)
    #env = FakeBatchedEnvironment(num_envs=1, players=[1, 2, 3, 4])

    # Push initial env state.
    initial_state = env.current_state()
    if batch_time:
      initial_state = [initial_state]
    send(initial_state)

    env_step = env.multi_step if batch_time else env.step

    while True:
      controllers = conn.recv()
      if controllers is None:
        send(None)  # signal end of outputs
        return
      send(env_step(controllers))

    # conn.close()
  except KeyboardInterrupt:
    # exit quietly without spamming stderr
    return
  except BrokenPipeError:
    # The other end closed the connection.
    return
  except Exception:
    send(EnvError(traceback.format_exc()))
    send(None)  # signal end of outputs
  finally:
    if env:
      env.stop()

class EnvError(Exception):
  pass

class AsyncEnvMP:
  """An asynchronous environment using multiprocessing."""

  def __init__(
      self,
      dolphin_kwargs: dict,
      num_envs: int = 0,  # zero means non-batched env
      slippi_ports: Optional[list[int]] = None,
      num_retries: int = 2,
      batch_time: bool = False,
      agent_names: list[tuple[str, str]] = [],
      singles_roles: Optional[list[tp.Optional[str]]] = None,
      **env_kwargs,
  ):
    context = mp.get_context('forkserver')
    self._parent_conn, child_conn = context.Pipe()
    self._recv = self._parent_conn.recv

    builder_kwargs = dict(
        num_envs=num_envs,
        dolphin_kwargs=dolphin_kwargs,
        slippi_ports=slippi_ports,
        num_retries=num_retries,
        agent_names=agent_names,
        singles_roles=singles_roles,
        **env_kwargs,
    )
    # self._stop = mp.Event()
    self._process = context.Process(
        name=f'_run_env({slippi_ports})',
        target=_run_env,
        args=(builder_kwargs, child_conn),
        kwargs=dict(batch_time=batch_time))
    self._process.start()
    
    # Performance instrumentation
    self._send_profiler = utils.Profiler()
    self._recv_profiler = utils.Profiler()

  def stop(self):
    self.begin_stop()
    self.ensure_stopped()

  def begin_stop(self):
    """Non-blocking stop."""
    if self._process is not None:
      # self._stop.set()
      try:
        self._parent_conn.send(None)
      except BrokenPipeError:
        pass

  def ensure_stopped(self):
    if self._process is None:
      return

    # The _run_env process might be blocked on pushing data into the Pipe.
    # To unblock it, we pull all the pending data from the pipe.
    while self._process.is_alive():
      try:
        # _run_env pushes None to signal execution has finished.
        if self._parent_conn.poll(1) and self._parent_conn.recv() is None:
          break
      except (ConnectionResetError, EOFError):
        break

    # logging.info('Joining process %d', self._process.pid)
    self._process.join()
    self._process.close()
    self._process = None

  # def __del__(self):
  #   self.stop()

  def send(self, controllers: tp.Union[Controllers, list[Controllers]]):
    try:
      with self._send_profiler:
        self._parent_conn.send(controllers)
    except BrokenPipeError:
      # Attempt to retrieve exception from pipe.
      while True:
        if not self._parent_conn.poll(1):
          break

        output = self._parent_conn.recv()
        if isinstance(output, Exception):
          raise output
        elif output is None:
          break

      # Fall back to raising a generic error message.
      raise EnvError("run_env process died")

  def recv(self) -> EnvOutput:
    # TODO: ensure that enough data has been pushed?
    try:
      with self._recv_profiler:
        output = self._recv()
    except ConnectionResetError as e:
      self.ensure_stopped()
      raise EnvError("run_env process died")
    if isinstance(output, Exception):
      # Maybe rebuild the environment and start over?
      raise output
    return output


def _select_ports(
    controllers: Controllers,
    ports: tp.Sequence[int],
) -> Controllers:
  return {port: controllers[port] for port in ports if port in controllers}


class _DoublesChunk:
  def __init__(self, env: AsyncEnvMP):
    self._env = env

  def send(self, controllers: Controllers):
    self._env.send(controllers)

  def recv(self) -> EnvOutput:
    return self._env.recv()

  def begin_stop(self):
    self._env.begin_stop()

  def ensure_stopped(self):
    self._env.ensure_stopped()


class _SinglesChunk:
  def __init__(self, left_env: AsyncEnvMP, right_env: AsyncEnvMP):
    self._left_env = left_env
    self._right_env = right_env

  def send(self, controllers: Controllers):
    left_controllers = _select_ports(controllers, (1, 2))
    right_controllers = _select_ports(controllers, (3, 4))

    self._left_env.send(left_controllers)
    self._right_env.send(right_controllers)

  def recv(self) -> EnvOutput:
    while True:
      left_output = self._left_env.recv()
      right_output = self._right_env.recv()

      if isinstance(left_output, list):
        if not isinstance(right_output, list):
          raise EnvError('Mismatched singles chunk outputs: list vs scalar')
        if len(left_output) != len(right_output):
          raise EnvError('Mismatched singles chunk batch lengths')
        if not left_output:
          # Empty batches can occur during resets; wait for the next payload.
          continue
        return [
            _merge_singles_env_outputs(lo, ro)
            for lo, ro in zip(left_output, right_output)
        ]

      if isinstance(right_output, list):
        raise EnvError('Mismatched singles chunk outputs: scalar vs list')

      return _merge_singles_env_outputs(left_output, right_output)

  def begin_stop(self):
    self._left_env.begin_stop()
    self._right_env.begin_stop()

  def ensure_stopped(self):
    self._left_env.ensure_stopped()
    self._right_env.ensure_stopped()

class AsyncBatchedEnvironmentMP:
  """A set of asynchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dolphin_kwargs: dict,
      num_steps: int = 0,
      inner_batch_size: int = 1,
      num_retries: int = 2,
      swap_ports: bool = True,
      enable_singles: bool = False, # Enable singles mode for half the envs
      singles_ratio: float = 0.5,
      agent_names: list[tuple[str, str]] = [],
  ):
    if num_envs % inner_batch_size != 0:
      raise ValueError(
          f'num_envs={num_envs} must be divisible by '
          f'inner_batch_size={inner_batch_size}')

    if swap_ports and inner_batch_size % 2 != 0:
      raise ValueError('swap_ports=True requires an even inner_batch_size.')

    self._total_batch_size = num_envs
    self._chunks: list[tp.Union[_SinglesChunk, _DoublesChunk]] = []
    layout = compute_mixed_mode_layout(
        total_envs=num_envs,
        singles_ratio=singles_ratio if enable_singles else 0.0,
        inner_batch_size=inner_batch_size,
    )
    self._outer_batch_size = len(layout.chunk_modes)
    self._inner_batch_size = inner_batch_size
    self._slice = lambda i, x: x[i * inner_batch_size:(i + 1) * inner_batch_size]

    self._dolphin_kwargs = dolphin_kwargs

    total_physical_envs = sum(layout.chunk_physical_sizes)
    slippi_ports = utils.find_open_udp_ports(total_physical_envs)

    port_cursor = 0
    for chunk_idx, mode in enumerate(layout.chunk_modes):
      agent_offset = chunk_idx * inner_batch_size
      env_agent_names = agent_names[agent_offset:agent_offset + inner_batch_size]

      chunk_size = layout.chunk_physical_sizes[chunk_idx]
      chunk_ports = slippi_ports[port_cursor:port_cursor + chunk_size]
      port_cursor += chunk_size

      if mode == 'singles':
        half = inner_batch_size
        left_ports = chunk_ports[:half]
        right_ports = chunk_ports[half:]

        left_env = AsyncEnvMP(
            dolphin_kwargs=dolphin_kwargs,
            num_envs=inner_batch_size,
            batch_time=(num_steps > 0),
            slippi_ports=left_ports,
            num_retries=num_retries,
            swap_ports=False,
            enable_singles=True,
            agent_names=env_agent_names,
            singles_roles=['left'] * inner_batch_size,
        )

        right_env = AsyncEnvMP(
            dolphin_kwargs=dolphin_kwargs,
            num_envs=inner_batch_size,
            batch_time=(num_steps > 0),
            slippi_ports=right_ports,
            num_retries=num_retries,
            swap_ports=False,
            enable_singles=True,
            agent_names=env_agent_names,
            singles_roles=['right'] * inner_batch_size,
        )

        self._chunks.append(_SinglesChunk(left_env, right_env))
      else:
        env = AsyncEnvMP(
            dolphin_kwargs=dolphin_kwargs,
            num_envs=inner_batch_size,
            batch_time=(num_steps > 0),
            slippi_ports=chunk_ports,
            num_retries=num_retries,
            swap_ports=swap_ports,
            enable_singles=False,
            agent_names=env_agent_names,
        )
        self._chunks.append(_DoublesChunk(env))

    self._num_steps = num_steps
    self._action_queue: list[Controllers] = []
    self._state_queue = collections.deque()
    self._num_in_transit = 1  # take into account initial state
    
    # Performance instrumentation
    self._env_wait_profiler = utils.Profiler()  # Time waiting for envs to respond
    self._serialization_in_profiler = utils.Profiler()  # Time spent processing received data
    self._serialization_out_profiler = utils.Profiler()  # Time spent preparing data to send
    self._total_receive_profiler = utils.Profiler()  # Total time spent in _receive()
    self._last_pop_time = None  # Timestamp of last pop() call
    self._controller_wait_times = []  # Time between pop() and push()
    self._step_counter = 0  # Counter for periodic logging
    self._log_frequency = 2000  # Log performance stats every N steps

  def qsize(self):
    return self._num_in_transit + len(self._state_queue)

  @property
  def num_steps(self) -> int:
    return self._num_steps or 1  # 0 means 1

  def stop(self):
    # First initiate stop asynchronously for all envs.
    for chunk in self._chunks:
      chunk.begin_stop()

    # Then ensure all envs are stopped.
    for chunk in self._chunks:
      chunk.ensure_stopped()

  def __del__(self):
    self.stop()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def _flush(self):
    # Returns a time-indexed list of controller dictionaries.
    with self._serialization_out_profiler:
      get_action = lambda i: utils.map_single_structure(
          lambda x: self._slice(i, x), self._action_queue)

      for i, chunk in enumerate(self._chunks):
        chunk.send(get_action(i))

    self._num_in_transit += len(self._action_queue)
    self._action_queue.clear()

  def push(self, controllers: Controllers):
    # Measure time gap from pop to push (model thinking time)
    if self._last_pop_time is not None:
        wait_time = time.perf_counter() - self._last_pop_time
        self._controller_wait_times.append(wait_time)
        self._last_pop_time = None
    
    self._step_counter += 1
    if self._step_counter >= self._log_frequency:
        self._log_performance_stats()
        self._step_counter = 0
    
    if self._num_steps == 0:
      with self._serialization_out_profiler:
        get_action = lambda i: utils.map_single_structure(
            lambda x: self._slice(i, x), controllers)
        for i, chunk in enumerate(self._chunks):
          chunk.send(get_action(i))
      self._num_in_transit += 1
      return

    self._action_queue.append(controllers)
    if len(self._action_queue) == self._num_steps:
      self._flush()

  def _receive(self):
    with self._total_receive_profiler:
      if self._num_steps == 0:
        with self._env_wait_profiler:
          outputs = [chunk.recv() for chunk in self._chunks]
        
        with self._serialization_in_profiler:
          output = utils.concat_nest_nt(outputs)
          self._state_queue.appendleft(output)
          self._num_in_transit -= 1
      else:
        with self._env_wait_profiler:
          batch_major = [chunk.recv() for chunk in self._chunks]
        
        with self._serialization_in_profiler:
          time_major = zip(*batch_major)
          for batch in time_major:
            self._state_queue.appendleft(utils.concat_nest_nt(batch))
            self._num_in_transit -= 1

  def pop(self) -> EnvOutput:
    if not self._state_queue:
      self._receive()
    result = self._state_queue.pop()
    self._last_pop_time = time.perf_counter()
    return result

  def peek_n(self, n: int) -> list[EnvOutput]:
    while len(self._state_queue) < n:
      self._receive()
    return utils.peek_deque(self._state_queue, n)

  def peek(self) -> EnvOutput:
    if not self._state_queue:
      self._receive()
    return self._state_queue[-1]

  def _log_performance_stats(self):
    """Log performance statistics periodically."""
    # Skip if we don't have enough data
    if (self._env_wait_profiler.num_calls == 0 or 
        not self._controller_wait_times):
        return
        
    # Calculate mean waiting time for model input
    '''controller_wait = sum(self._controller_wait_times) / len(self._controller_wait_times)
    self._controller_wait_times = []  # Reset list
    
    # Calculate average serialization times from individual envs
    env_send_times = [env._send_profiler.mean_time() * 1000 for env in self._envs 
                     if env._send_profiler.num_calls > 0]
    env_recv_times = [env._recv_profiler.mean_time() * 1000 for env in self._envs 
                     if env._recv_profiler.num_calls > 0]
    
    avg_env_send = sum(env_send_times) / len(env_send_times) if env_send_times else 0
    avg_env_recv = sum(env_recv_times) / len(env_recv_times) if env_recv_times else 0
    
    # Log the performance stats
    logging.info(
        f"Env performance stats (avg ms, queue: {len(self._state_queue)}, in_transit: {self._num_in_transit}):\n"
        f"  Env wait time: {self._env_wait_profiler.mean_time() * 1000:.2f}\n"
        f"  Recv serialization: {self._serialization_in_profiler.mean_time() * 1000:.2f}\n"
        f"  Send serialization: {self._serialization_out_profiler.mean_time() * 1000:.2f}\n"
        f"  Total receive time: {self._total_receive_profiler.mean_time() * 1000:.2f}\n"
        f"  Env send time: {avg_env_send:.2f}\n"
        f"  Env recv time: {avg_env_recv:.2f}\n"
        f"  Controller wait time: {controller_wait * 1000:.2f}"
    )
    
    # Periodically reset profilers (every 10 logs)'''
    if self._step_counter % (self._log_frequency * 10) == 0:
        self._reset_profilers()
  
  def _reset_profilers(self):
    """Reset all profilers to avoid accumulating data over very long periods."""
    logging.info("Resetting performance profilers")
    self._env_wait_profiler = utils.Profiler()
    self._serialization_in_profiler = utils.Profiler()
    self._serialization_out_profiler = utils.Profiler()
    self._total_receive_profiler = utils.Profiler()
    
    # Reset profilers in child environments
    for chunk in self._chunks:
        if isinstance(chunk, _SinglesChunk):
            chunk._left_env._send_profiler = utils.Profiler()
            chunk._left_env._recv_profiler = utils.Profiler()
            chunk._right_env._send_profiler = utils.Profiler()
            chunk._right_env._recv_profiler = utils.Profiler()
        else:
            chunk._env._send_profiler = utils.Profiler()
            chunk._env._recv_profiler = utils.Profiler()

reified_game = utils.reify_tuple_type(Game)

class FakeBatchedEnvironment:
  def __init__(
      self,
      num_envs: int,
      players: tp.Collection[int],
  ):
    game = utils.map_nt(
        lambda t: np.full([num_envs], 0, dtype=t), reified_game)
    game.stage[:] = Stage.FINAL_DESTINATION.value  # make the stage valid
    self._dummy_output = EnvOutput(
        gamestates={p: game for p in players},
        needs_reset=np.full([num_envs], False),
    )
    self.num_steps = 1
    self._output_queue = collections.deque()
    self._output_queue.append(self._dummy_output)

  def stop(self):
    pass

  def current_state(self) -> EnvOutput:
    return self._dummy_output

  def pop(self) -> EnvOutput:
    return self._output_queue.popleft()

  def push(self, controllers: Controllers):
    # TODO: increment frame counter in the gamestates
    del controllers
    self._output_queue.append(self._dummy_output)

  def step(self, controllers: Controllers):
    self.push(controllers)
    return self.pop()

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    return [self.step(c) for c in controllers]

  def peek(self) -> EnvOutput:
    return self._output_queue[0]

class ReplayBatchedEnvironment:
  def __init__(
      self,
      num_envs: int,
      players: tp.Collection[int],
  ):
    self.batch_size = num_envs
    self.data_source = data.toy_data_source(
        batch_size=1, unroll_length=1, extra_frames=0)
    self.players = players
    self.num_steps = 1
    self._output_queue = collections.deque()
    self._push_output()

  def _push_output(self):
    batch, _ = next(self.data_source)

    gamestates = {}
    for i, p in enumerate(self.players):
      game = batch.frames.state_action.state
      swap = i % 2 == 1
      if swap:
        game = data.swap_players(game)
      gamestates[p] = game

    output = EnvOutput(
        gamestates=gamestates,  # [B=1, T=1]
        needs_reset=batch.frames.is_resetting,  # [B=1, T=1]
    )
    output = utils.map_nt(np.squeeze, output)  # []
    output = utils.map_nt(lambda x: np.tile(x, [self.batch_size]), output)

    self._output_queue.append(output)

  def stop(self):
    pass

  def pop(self) -> EnvOutput:
    return self._output_queue.popleft()

  def push(self, controllers: Controllers):
    del controllers
    self._push_output()

  def step(self, controllers: Controllers):
    self.push(controllers)
    return self.pop()

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    return [self.step(c) for c in controllers]

  def peek(self) -> EnvOutput:
    return self._output_queue[0]
