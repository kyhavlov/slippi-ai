import atexit
import collections
import dataclasses
import itertools
import logging
import json
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from concurrent import futures
from typing import (
    Any, Callable, Iterable, List, Optional, Set, Tuple, Iterator, NamedTuple,
    Union,
)
import zlib

import numpy as np
import pyarrow
import pyarrow.parquet as pq

import melee

from slippi_ai import reward, utils, nametags, paths
from slippi_db import file_layout
from slippi_ai.types import Game, game_array_to_nt, Controller

class PlayerMeta(NamedTuple):
  character: int
  name: str
  team: int

  @classmethod
  def from_metadata(cls, player_meta: dict, raw: str) -> 'PlayerMeta':
    team = 0 if 'team' not in player_meta else player_meta['team']

    return cls(
        character=player_meta['character'],
        name=nametags.name_from_metadata(player_meta, raw=raw),
        team=team,
    )

class ReplayMeta(NamedTuple):
  p0: PlayerMeta
  p1: PlayerMeta
  p2: PlayerMeta
  p3: PlayerMeta
  stage: int
  slp_md5: str
  is_singles: bool = False

  @classmethod
  def from_metadata(cls, metadata: dict) -> 'ReplayMeta':
    raw = metadata['raw']
    p0=PlayerMeta.from_metadata(metadata['players'][0], raw)
    p1=PlayerMeta.from_metadata(metadata['players'][1], raw)
    if len(metadata['players']) == 4:
      p2=PlayerMeta.from_metadata(metadata['players'][2], raw)
      p3=PlayerMeta.from_metadata(metadata['players'][3], raw)
      is_singles = False
    else:
      p2 = PlayerMeta(character=0, name='', team=0)
      p3 = PlayerMeta(character=0, name='', team=0)
      is_singles = True

    return cls(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        stage=metadata['stage'],
        slp_md5=metadata['slp_md5'],
        is_singles=is_singles)

class ReplayInfo(NamedTuple):
  path: str
  # We use empty tuple instead of None to play nicely with Tensorflow.
  main_player_index: int
  teammate_index: int
  main_player_name: str
  meta: Union[ReplayMeta, Tuple[()]] = ()
  opponent_order: Tuple[int, int] = (2, 3)

  @property
  def main_player(self) -> PlayerMeta:
    if self.main_player_index == 0:
      return self.meta.p0
    elif self.main_player_index == 1:
      return self.meta.p1
    elif self.main_player_index == 2:
      return self.meta.p2
    return self.meta.p3

class ChunkMeta(NamedTuple):
  start: int
  end: int
  info: ReplayInfo

class Chunk(NamedTuple):
  states: Game
  meta: ChunkMeta

# Action = TypeVar('Action')
Action = Controller

class StateAction(NamedTuple):
  state: Game
  # The action could actually be an "encoded" action type,
  # which might discretize certain components of the controller
  # such as the sticks and shoulder. Unfortunately NamedTuples can't be
  # generic. We could use a dataclass instead, but TF can't trace them.
  # Note that this is the action taken on the _previous_ frame.
  action: Action

  # Encoded name
  name: int

class Frames(NamedTuple):
  state_action: StateAction
  is_resetting: bool
  # The reward will have length one less than the states and actions.
  reward: np.float32

class Batch(NamedTuple):
  frames: Frames
  count: int  # For reproducing batches
  meta: ChunkMeta


class ReplayTask(NamedTuple):
  replay: ReplayInfo
  epoch_progress: float


def _split_replays_by_mode(replays: Iterable[ReplayInfo]) -> Tuple[list[ReplayInfo], list[ReplayInfo], list[ReplayInfo]]:
  singles: list[ReplayInfo] = []
  doubles: list[ReplayInfo] = []
  unknown: list[ReplayInfo] = []

  for replay in replays:
    meta = getattr(replay, 'meta', ())
    is_singles = getattr(meta, 'is_singles', None)
    if is_singles is None:
      unknown.append(replay)
    elif is_singles:
      singles.append(replay)
    else:
      doubles.append(replay)

  return singles, doubles, unknown


def _cycle_items(items: Iterable[ReplayInfo]) -> Optional[Iterator[ReplayInfo]]:
  items = tuple(items)
  if not items:
    return None
  return itertools.cycle(items)


def _round_robin_iterators(*iterables: Iterable[ReplayInfo]) -> Optional[Iterator[ReplayInfo]]:
  non_empty = [it for it in iterables if it is not None]
  if not non_empty:
    return None
  return utils.interleave(*non_empty)


def _distribute_unknown_between_modes(
    singles: Iterable[ReplayInfo],
    doubles: Iterable[ReplayInfo],
    unknown: Iterable[ReplayInfo],
) -> Tuple[list[ReplayInfo], list[ReplayInfo]]:
  singles_list = list(singles)
  doubles_list = list(doubles)
  unknown_list = list(unknown)

  for idx, replay in enumerate(unknown_list):
    if len(singles_list) <= len(doubles_list):
      singles_list.append(replay)
    else:
      doubles_list.append(replay)

  return singles_list, doubles_list


def _character_cycle(replays: Iterable[ReplayInfo]) -> Optional[Iterator[ReplayInfo]]:
  groups: dict[int, list[ReplayInfo]] = collections.defaultdict(list)

  for replay in replays:
    try:
      character = int(replay.main_player.character)
    except Exception:
      continue
    groups[character].append(replay)

  valid_groups = [tuple(entries) for entries in groups.values() if entries]
  if len(valid_groups) <= 1:
    return None

  iterables = [itertools.cycle(group) for group in valid_groups]
  return utils.interleave(*iterables)


def _character_counts_for_logging(replays: Iterable[ReplayInfo]) -> dict[str, int]:
  counts: collections.Counter = collections.Counter()

  for replay in replays:
    try:
      character = int(replay.main_player.character)
    except Exception:
      continue
    try:
      name = melee.Character(character).name.lower()
    except ValueError:
      name = str(character)
    counts[name] += 1

  return dict(counts)


class _ModeSampler:
  def __init__(
      self,
      replays: Iterable[ReplayInfo],
      *,
      label: str,
      balance_ratio: float,
  ):
    self.replays: Tuple[ReplayInfo, ...] = tuple(replays)
    self.available = bool(self.replays)
    self._raw_iter = _cycle_items(self.replays)
    self._balanced_iter = _character_cycle(self.replays) if self.replays else None
    self.has_balanced = self._balanced_iter is not None
    self._balance_ratio = max(0.0, min(balance_ratio, 1.0))
    self._balance_budget = 0.0
    if self.has_balanced and label != 'global' and self._balance_ratio > 0:
      logging.info('Character balance counts [%s]: %s',
                   label, _character_counts_for_logging(self.replays))
    self._toggle = False

  def next(self, use_balanced: bool) -> ReplayInfo:
    if self._raw_iter is None:
      raise RuntimeError('No replays available for sampling.')

    if (use_balanced and self._balanced_iter is not None and
        self._balance_ratio > 0):
      if self._balance_ratio >= 1.0:
        return next(self._balanced_iter)

      self._balance_budget += self._balance_ratio
      if self._balance_budget >= 1.0:
        self._balance_budget -= 1.0
        return next(self._balanced_iter)

    return next(self._raw_iter)

class _ReplaySampler:
  """Generates replay sequences with optional balancing constraints."""

  def __init__(
      self,
      replays: Iterable[ReplayInfo],
      *,
      balance_characters: bool = False,
      balance_singles_doubles: bool = False,
      character_balance_ratio: float = 0.5,
  ):
    replays = tuple(replays)
    if not replays:
      raise ValueError('ReplaySampler requires at least one replay.')

    self._replays = replays
    self._balance_characters = balance_characters
    self._balance_singles_doubles = balance_singles_doubles
    self._character_balance_ratio = (
        max(0.0, min(character_balance_ratio, 1.0))
        if balance_characters else 0.0)

    singles, doubles, unknown = _split_replays_by_mode(replays)
    self._singles = singles
    self._doubles = doubles
    self._unknown = unknown

    singles_ext, doubles_ext = _distribute_unknown_between_modes(
        self._singles, self._doubles, self._unknown)
    self._singles_sampler = _ModeSampler(
        singles_ext,
        label='singles',
        balance_ratio=self._character_balance_ratio)
    self._doubles_sampler = _ModeSampler(
        doubles_ext,
        label='doubles',
        balance_ratio=self._character_balance_ratio)

    self._mode_toggle = False
    self._mode_balance_available = (
        self._balance_singles_doubles
        and self._singles_sampler.available
        and self._doubles_sampler.available)
    self._warned_mode_balance = False

    self._global_sampler = _ModeSampler(
        self._replays,
        label='global',
        balance_ratio=self._character_balance_ratio)

  def next(self) -> ReplayInfo:
    if self._mode_balance_available:
      sampler = (self._singles_sampler if not self._mode_toggle
                 else self._doubles_sampler)
      self._mode_toggle = not self._mode_toggle
      return sampler.next(self._balance_characters)

    if self._balance_singles_doubles and not self._warned_mode_balance:
      logging.info(
          'Falling back to dataset distribution; unable to balance '
          'singles/doubles (singles=%d, doubles=%d).',
          len(self._singles_sampler.replays), len(self._doubles_sampler.replays))
      self._warned_mode_balance = True

    if self._balance_characters and self._global_sampler.has_balanced:
      replay = self._global_sampler.next(True)
    else:
      replay = self._global_sampler.next(False)
    return replay

  def iterator(self) -> Iterator[ReplayInfo]:
    while True:
      yield self.next()


def replay_stream(
    replays: Iterable[ReplayInfo],
    *,
    balance_characters: bool = False,
    balance_singles_doubles: bool = False,
    character_balance_ratio: float = 0.5,
) -> Iterator[ReplayInfo]:
  """Return an infinite iterator over replays respecting balancing options."""
  sampler = _ReplaySampler(
      list(replays),
      balance_characters=balance_characters,
      balance_singles_doubles=balance_singles_doubles,
      character_balance_ratio=character_balance_ratio,
  )
  return sampler.iterator()

def _charset(chars: Optional[Iterable[melee.Character]]) -> Set[int]:
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)


@dataclasses.dataclass
class DatasetConfig:
  data_dir: Optional[str] = None  # required
  meta_path: Optional[str] = None
  test_ratio: float = 0.1
  # comma-separated lists of characters, or "all"
  allowed_characters: str = 'all'
  allowed_opponents: str = 'all'
  swap: bool = True  # yield both p2/p3 opponent permutations
  seed: int = 0
  include2v2: bool = True

def replays_from_meta(config: DatasetConfig) -> List[ReplayInfo]:
  replays = []

  with open(config.meta_path) as f:
    meta_rows: list[dict] = json.load(f)

  allowed_characters = _charset(chars_from_string(config.allowed_characters))
  allowed_opponents = _charset(chars_from_string(config.allowed_opponents))

  banned_counts = collections.Counter()

  invalid_team_id_count = 0

  for row in meta_rows:
    replay_meta = ReplayMeta.from_metadata(row)
    replay_path = file_layout.resolve_parquet_path(config.data_dir, replay_meta.slp_md5)

    # for singles games, generate two replays (one for each player). 
    # each replay will have the self player as p0 and the opponent randomized
    # as either p2 or p3 seeded by the replay_meta.slp_md5 value
    if replay_meta.is_singles:
      players = [replay_meta.p0, replay_meta.p1]

      # Promote each player to the p0 slot while keeping teammate empty (p1) and
      # the opponent in the remaining visible slot after swapping.
      for player_index, player in enumerate(players):
        opponent = players[1 - player_index]

        if player.character not in allowed_characters:
          continue

        if opponent.character not in allowed_opponents:
          continue

        if nametags.is_banned_name(player.name):
          banned_counts[player.name] += 1
          continue

        empty_player = PlayerMeta(character=0, name='', team=0)

        if player_index == 0:
          main_index = 0
        else:
          main_index = 2

        teammate_index = 1
        other_ports = [i for i in range(4) if i not in (main_index, teammate_index)]
        assert len(other_ports) == 2

        hash_val = int(replay_meta.slp_md5[-1], 16)
        opponent_order = tuple(other_ports if hash_val % 2 == 0 else reversed(other_ports))
        opponent_orders = [opponent_order]
        if config.swap:
          swapped = tuple(reversed(opponent_order))
          if swapped != opponent_order:
            opponent_orders.append(swapped)

        for order in opponent_orders:
          meta_slots = [empty_player, empty_player, empty_player, empty_player]
          meta_slots[main_index] = player
          meta_slots[order[0]] = opponent
          meta_slots[order[1]] = empty_player

          meta = ReplayMeta(
              p0=meta_slots[0],
              p1=meta_slots[1],
              p2=meta_slots[2],
              p3=meta_slots[3],
              stage=replay_meta.stage,
              slp_md5=replay_meta.slp_md5,
              is_singles=True,
          )

          replays.append(ReplayInfo(
              replay_path,
              main_player_index=main_index,
              teammate_index=teammate_index,
              main_player_name=player.name,
              meta=meta,
              opponent_order=order,
          ))

      continue

    # append a replay for each player index
    for player_index in range(4):
      players = [replay_meta.p0, replay_meta.p1, replay_meta.p2, replay_meta.p3]

      if players[player_index].character not in allowed_characters:
        continue

      # check whether any of the other characters are not allowed
      if any(p.character not in allowed_opponents for i, p in enumerate(players) if i != player_index):
        continue

      if nametags.is_banned_name(players[player_index].name):
        banned_counts[players[player_index].name] += 1
        continue

      # make sure the player+character combo is in REQUIRED_PLAYERS
      '''is_required_player = False
      for codes, character in nametags.REQUIRED_PLAYERS:
        if players[player_index].name in codes and players[player_index].character == character.value:
          is_required_player = True
          break

      if not is_required_player:
        continue'''

      team_id = players[player_index].team
      # find teammate index
      teammate_index = next((i for i, p in enumerate(players) if p.team == team_id and i != player_index), -1)

      if teammate_index != -1:
        other_ports = tuple(i for i in range(4) if i not in (player_index, teammate_index))
        assert len(other_ports) == 2
        info = ReplayInfo(
            replay_path,
            player_index,
            teammate_index,
            players[player_index].name,
            replay_meta,
            other_ports,
        )
        replays.append(info)
        if config.swap:
          swapped = tuple(reversed(other_ports))
          if swapped != other_ports:
            replays.append(info._replace(opponent_order=swapped))
      else:
        invalid_team_id_count += 1

      #print("added replay", row['name'], player_index, players[player_index].name, players[player_index].character, teammate_index)

  #print("invalid team id game count (3v1?): ", invalid_team_id_count)
  #print('Banned names:', banned_counts)

  return replays

def train_test_split(
    config: DatasetConfig,
) -> Tuple[List[ReplayInfo], List[ReplayInfo]]:
  replays: list[ReplayInfo] = []

  if config.meta_path is not None:
    replays = replays_from_meta(config)

    # Ensure every replay referenced in metadata exists on disk.
    missing = [info.path for info in replays if not os.path.exists(info.path)]
    if missing:
      raise FileNotFoundError(
          f"Missing {len(missing)} parquet files referenced in metadata; "
          f"sample: {missing[:3]}")
  else:
    raise ValueError("Please provide a metadata file.")

  # TODO: stable partition
  rng = random.Random(config.seed)
  rng.shuffle(replays)
  split_idx = int(len(replays) * config.test_ratio)
  return replays[split_idx:], replays[:split_idx]

name_to_character = {c.name.lower(): c for c in melee.Character}

def chars_from_string(chars: str) -> Optional[List[melee.Character]]:
  if chars == 'all':
    return None
  chars = chars.split(',')
  return [name_to_character[c] for c in chars]


def game_len(game: Game):
  return len(game.stage)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(
      self,
      source: Iterator[ReplayInfo],
      unroll_length: int,
      overlap: int = 1,
      compressed: bool = True,
      game_filter: Optional[Callable[[Game], bool]] = None,
  ):
    self.source = source
    self.compressed = compressed
    self.unroll_length = unroll_length
    self.overlap = overlap
    self.game_filter = game_filter or (lambda _: True)

    self.game: Game = None
    self.frame: int = None
    self.info: ReplayInfo = None

  def load_game(self, info: ReplayInfo) -> Game:
    game = read_table(info.path, compressed=self.compressed)
    # print('pre-swap: ', id(game.p0), id(game.p1), id(game.p2), id(game.p3), info.main_player_index, info.teammate_index)
    game = swap_players(game, info)
    # print('post-swap: ', id(game.p0), id(game.p1), id(game.p2), id(game.p3))

    # check for nans in any player data
    '''for i, player in enumerate([game.p0, game.p1, game.p2, game.p3]):
      fields = [player.percent, player.facing, player.x, player.y, player.action,
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
          print(f'NAN in field {j} for player {i} in {info.path}: ', nan_locs)
          was_nan = True

      assert not was_nan'''

    return game

  def find_game(self):
    while True:
      info = next(self.source)
      game = self.load_game(info)
      if game_len(game) < self.unroll_length:
        continue
      if not self.game_filter(game):
        continue
      break
    self.game = game
    self.frame = 0
    self.info = info

  def grab_chunk(self) -> Chunk:
    """Grabs a chunk from a trajectory."""
    # TODO: write a unit test for this

    needs_reset = (
        self.game is None or
        self.frame + self.unroll_length > game_len(self.game))

    if needs_reset:
      self.find_game()

    start = self.frame
    end = start + self.unroll_length
    slice = lambda a: a[start:end]
    # faster than tree.map_structure
    states = utils.map_nt(slice, self.game)
    self.frame = end - self.overlap

    return Chunk(states, ChunkMeta(start, end, self.info))

# swap players to the ports specified by the given ReplayInfo
def swap_players(game: Game, info: ReplayInfo) -> Game:
  # make a list of the ports specified for opponents
  if info.opponent_order is not None:
    opponent_ports = info.opponent_order
  else:
    opponent_ports = tuple(i for i in range(4)
                           if i not in (info.main_player_index, info.teammate_index))

  p0 = getattr(game, f'p{info.main_player_index}')
  p1 = getattr(game, f'p{info.teammate_index}')
  p2 = getattr(game, f'p{opponent_ports[0]}')
  p3 = getattr(game, f'p{opponent_ports[1]}')
  return game._replace(p0=p0, p1=p1, p2=p2, p3=p3, stage=game.stage)

def read_table(path: str, compressed: bool) -> Game:
  if compressed:
    with open(path, 'rb') as f:
      contents = f.read()
    contents = zlib.decompress(contents)
    reader = pyarrow.BufferReader(contents)
    table = pq.read_table(reader)
  else:
    table = pq.read_table(path)

  game_struct = table['root'].combine_chunks()
  return game_array_to_nt(game_struct)

class DataSource:
  def __init__(
      self,
      replays: List[ReplayInfo],
      compressed: bool = True,
      batch_size: int = 64,
      unroll_length: int = 64,
      extra_frames: int = 1,
      damage_ratio: float = 0.01,
      # None means all allowed.
      allowed_characters: Optional[list[melee.Character]] = None,
      allowed_opponents: Optional[list[melee.Character]] = None,
      balance_characters: bool = False,
      balance_singles_doubles: bool = False,
      character_balance_ratio: float = 0.5,
      name_map: Optional[dict[str, int]] = None,
      num_workers: int = 1,
      replay_queue: Optional[queue.Queue] = None,
  ):
    self.replays = replays
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.damage_ratio = damage_ratio
    self.compressed = compressed
    self.batch_counter = 0
    self.balance_characters = balance_characters
    self.balance_singles_doubles = balance_singles_doubles
    self.character_balance_ratio = max(0.0, min(character_balance_ratio, 1.0))

    self._executor: Optional[futures.ThreadPoolExecutor] = None
    self._num_workers = max(1, num_workers)

    self.replay_counter = 0
    self._latest_epoch = 0.0
    self._using_queue = replay_queue is not None
    self._total_replays = len(self.replays)
    if replay_queue is not None:
      replays_iter: Iterator[ReplayInfo] = _QueueReplayIterator(self, replay_queue)
    else:
      replays_iter = self.iter_replays()
    if self._num_workers > 1:
      replays_iter = _ThreadSafeIterator(replays_iter)
      self._executor = futures.ThreadPoolExecutor(max_workers=self._num_workers)
    self.managers = [
        TrajectoryManager(
            replays_iter,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            compressed=compressed,
            game_filter=self.is_allowed)
        for _ in range(batch_size)]

    self.allowed_characters = _charset(allowed_characters)
    self.allowed_opponents = _charset(allowed_opponents)
    self.name_map = name_map or {}
    self.encode_name = nametags.name_encoder(self.name_map)

    self._replay_sampler: Optional[_ReplaySampler] = None
    if replay_queue is None:
      self._ensure_sampler()

  def _ensure_sampler(self):
    if getattr(self, '_replay_sampler', None) is None:
      balance_characters = getattr(self, 'balance_characters', False)
      balance_modes = getattr(self, 'balance_singles_doubles', False)
      balance_ratio = getattr(self, 'character_balance_ratio', 0.5)
      self._replay_sampler = _ReplaySampler(
          getattr(self, 'replays', []),
          balance_characters=balance_characters,
          balance_singles_doubles=balance_modes,
          character_balance_ratio=balance_ratio,
      )

  def iter_replays(self) -> Iterator[ReplayInfo]:
    self._ensure_sampler()
    assert self._replay_sampler is not None
    sampler_iter = self._replay_sampler.iterator()
    for replay in sampler_iter:
      self.replay_counter += 1
      yield replay

  def is_allowed(self, game: Game) -> bool:
    # TODO: handle Zelda/Sheik transformation
    return True
    return (
        game.p0.character[0] in self.allowed_characters
        and
        game.p1.character[0] in self.allowed_opponents)

  def process_game(
      self, game: Game, name_code: int, needs_reset: bool) -> Frames:
    game_length = game_len(game)
    assert game_length == self.chunk_size
    # Rewards could be deferred to the learner.
    rewards = reward.compute_rewards(game, damage_ratio=self.damage_ratio)
    name_codes = np.full([game_length], name_code, np.int32)
    state_action = StateAction(game, game.p0.controller, name_codes)
    is_resetting = np.full([game_length], False)
    is_resetting[0] = needs_reset
    return Frames(
        state_action=state_action, reward=rewards, is_resetting=is_resetting)

  def process_batch(self, chunks: list[Chunk]) -> Batch:
    batches: List[Batch] = []

    for chunk in chunks:
      name_code = self.encode_name(chunk.meta.info.main_player.name)
      needs_reset = chunk.meta.start == 0
      batches.append(Batch(
          frames=self.process_game(chunk.states, name_code, needs_reset),
          count=self.batch_counter,
          meta=chunk.meta))

    return utils.batch_nest_nt(batches)

  def __next__(self) -> Tuple[Batch, float]:
    if self._executor is None:
      chunks = [m.grab_chunk() for m in self.managers]
    else:
      futures_list = [self._executor.submit(m.grab_chunk) for m in self.managers]
      chunks = [future.result() for future in futures_list]
    batch: Batch = self.process_batch(chunks)
    if self._using_queue and self._total_replays > 0:
      epoch = self._latest_epoch
    else:
      epoch = self.replay_counter / max(1, len(self.replays))
    self.batch_counter += 1
    assert batch.frames.state_action.state.stage.shape[-1] == self.chunk_size
    assert batch.frames.reward.shape[-1] == self.chunk_size - 1
    return batch, epoch

  def close(self):
    if self._executor is not None:
      self._executor.shutdown(wait=True)
      self._executor = None

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass

def produce_batches(
    process_id: int,
    data_source_kwargs,
    batch_queue,
    task_queue,
):
  data_source = DataSource(replay_queue=task_queue, **data_source_kwargs)
  try:
    while True:
      batch_epoch = next(data_source)
      batch_queue.put((process_id, *batch_epoch))
  except StopIteration:
    pass
  finally:
    batch_queue.put((process_id, None, None))


def _split_batch_sizes(total: int, num_shards: int) -> list[int]:
  if num_shards <= 0:
    return []
  num_shards = min(total, num_shards)
  base = total // num_shards
  rem = total % num_shards
  sizes = []
  for idx in range(num_shards):
    shard = base + (1 if idx < rem else 0)
    if shard <= 0:
      continue
    sizes.append(shard)
  return sizes

class DataSourceMP:
  def __init__(self, buffer=2, num_processes: int = 1, **kwargs):
    if 'batch_size' not in kwargs:
      raise ValueError('batch_size must be provided to DataSourceMP.')
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.batch_size = kwargs['batch_size']
    requested_processes = max(1, num_processes)
    shard_sizes = _split_batch_sizes(self.batch_size, requested_processes)

    self._all_replays: list[ReplayInfo] = list(kwargs.get('replays', []) or [])
    if not self._all_replays:
      raise ValueError('DataSourceMP requires a non-empty replay list.')

    self.batch_queue = mp.Queue(buffer)
    task_buffer = max(1, buffer * len(shard_sizes) * 2)
    self.task_queue = mp.Queue(task_buffer)
    self._processes: list[mp.Process] = []
    self._process_sizes: dict[int, int] = {}
    self._closed = False
    self._global_batch_counter = 0
    self._pending: dict[int, list[Tuple[Batch, float]]] = {}
    self._scheduler_stop = threading.Event()
    self._balance_characters = kwargs.get('balance_characters', False)
    self._balance_singles_doubles = kwargs.get('balance_singles_doubles', False)
    self._character_balance_ratio = max(
        0.0, min(kwargs.get('character_balance_ratio', 0.5), 1.0))
    self._epoch_base = 0.0
    self._total_replays = len(self._all_replays)
    self._replay_sampler = _ReplaySampler(
        self._all_replays,
        balance_characters=self._balance_characters,
        balance_singles_doubles=self._balance_singles_doubles,
        character_balance_ratio=self._character_balance_ratio,
    )
    self._scheduler_iter = self._replay_sampler.iterator()

    replays = kwargs.get('replays', None)
    num_shards = len(shard_sizes)
    for idx, shard_size in enumerate(shard_sizes):
      proc_kwargs = dict(kwargs)
      proc_kwargs['batch_size'] = shard_size
      if replays is not None:
        proc_kwargs['replays'] = replays
      process = mp.Process(
          target=produce_batches,
          args=(idx, proc_kwargs, self.batch_queue, self.task_queue))
      process.start()
      self._processes.append(process)
      self._process_sizes[idx] = shard_size

    if not self._processes:
      raise ValueError('Failed to start any data loader processes.')

    if sum(self._process_sizes.values()) != self.batch_size:
      raise ValueError('Loader shard sizes do not sum to batch size.')

    self._scheduler_thread = threading.Thread(
        target=self._scheduler_loop, name='replay-scheduler', daemon=True)
    self._scheduler_thread.start()

    atexit.register(self.close)

  def __next__(self) -> Tuple[Batch, float]:
    micro_batches: list[Batch] = []
    weighted_epochs = 0.0
    total = 0
    seen_processes: set[int] = set()

    for proc_id, shard_size in self._process_sizes.items():
      pending = self._pending.get(proc_id, [])
      if pending:
        batch, epoch = pending.pop(0)
        micro_batches.append(batch)
        weighted_epochs += epoch * shard_size
        total += shard_size
        seen_processes.add(proc_id)
        if not pending:
          self._pending.pop(proc_id, None)

    while total < self.batch_size:
      proc_id, batch, epoch = self.batch_queue.get()
      if proc_id not in self._process_sizes:
        continue
      if batch is None:
        continue
      shard_size = self._process_sizes[proc_id]
      actual_size = batch.frames.state_action.state.stage.shape[0]
      if actual_size != shard_size:
        raise ValueError('Unexpected micro batch size from loader process.')
      if total + shard_size > self.batch_size:
        self._pending.setdefault(proc_id, []).append((batch, epoch))
        continue
      micro_batches.append(batch)
      weighted_epochs += epoch * shard_size
      total += shard_size
      seen_processes.add(proc_id)

    if total != self.batch_size:
      raise ValueError('Aggregated batch size mismatch.')

    combined_batch: Batch = utils.concat_nest_nt(micro_batches)
    self._global_batch_counter += 1
    if isinstance(combined_batch.count, np.ndarray):
      combined_batch.count.fill(self._global_batch_counter)

    combined_epoch = weighted_epochs / float(self.batch_size)
    return combined_batch, combined_epoch

  def _scheduler_loop(self):
    while not self._scheduler_stop.is_set():
      epoch_replays = self._build_epoch_replays()
      if not epoch_replays:
        time.sleep(0.1)
        continue
      epoch_start = self._epoch_base
      total = len(epoch_replays)
      for idx, replay in enumerate(epoch_replays, start=1):
        if self._scheduler_stop.is_set():
          break
        progress = epoch_start + idx / total
        task = ReplayTask(replay, progress)
        while not self._scheduler_stop.is_set():
          try:
            self.task_queue.put(task, timeout=0.1)
            break
          except queue.Full:
            continue
      self._epoch_base += 1.0

  def _build_epoch_replays(self) -> list[ReplayInfo]:
    return [next(self._scheduler_iter) for _ in range(self._total_replays)]

  def close(self):
    if self._closed:
      return
    self._closed = True
    self._scheduler_stop.set()
    if hasattr(self, '_scheduler_thread') and self._scheduler_thread.is_alive():
      self._scheduler_thread.join(timeout=0.1)
    try:
      while True:
        self.batch_queue.get_nowait()
    except queue.Empty:
      pass
    finally:
      self.batch_queue.close()
    try:
      while True:
        self.task_queue.get_nowait()
    except queue.Empty:
      pass
    finally:
      for _ in range(self.batch_size * 2):
        try:
          self.task_queue.put_nowait(None)
        except queue.Full:
          break
      self.task_queue.close()
    for process in self._processes:
      if process.is_alive():
        process.terminate()
      process.join()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass

@dataclasses.dataclass
class DataConfig:
  batch_size: int = 32
  unroll_length: int = 64
  damage_ratio: float = 0.01
  compressed: bool = True
  in_parallel: bool = True
  balance_characters: bool = False
  balance_singles_doubles: bool = False
  character_balance_ratio: float = 0.5
  num_workers: int = 1
  prefetch_buffer: int = 0
  loader_processes: int = 1

def make_source(
    in_parallel: bool,
    **kwargs):
  prefetch_buffer = kwargs.pop('prefetch_buffer', 0)
  loader_processes = kwargs.pop('loader_processes', 1)
  constructor = DataSourceMP if in_parallel else DataSource
  if constructor is DataSourceMP:
    source = constructor(num_processes=loader_processes, **kwargs)
  else:
    source = constructor(**kwargs)
  if prefetch_buffer:
    source = PrefetchDataIterator(source, maxsize=prefetch_buffer)
  return source

def toy_data_source(**kwargs) -> DataSource:
  dataset_config = DatasetConfig(
      data_dir=paths.TOY_DATA_DIR,
      meta_path=paths.TOY_META_PATH,
  )
  return DataSource(
      replays=replays_from_meta(dataset_config),
      compressed=True,
      **kwargs,
  )


class _ThreadSafeIterator:

  def __init__(self, iterator: Iterator[ReplayInfo]):
    self._iterator = iterator
    self._lock = threading.Lock()

  def __iter__(self) -> '_ThreadSafeIterator':
    return self

  def __next__(self) -> ReplayInfo:
    with self._lock:
      return next(self._iterator)


class _QueueReplayIterator:

  def __init__(self, data_source: 'DataSource', replay_queue: queue.Queue):
    self._data_source = data_source
    self._queue = replay_queue

  def __iter__(self) -> '_QueueReplayIterator':
    return self

  def __next__(self) -> ReplayInfo:
    task = self._queue.get()
    if task is None:
      raise StopIteration
    replay, epoch_progress = task
    if replay is None:
      raise StopIteration
    self._data_source.replay_counter += 1
    self._data_source._latest_epoch = epoch_progress
    return replay


class PrefetchDataIterator:

  def __init__(self, iterator, maxsize: int = 1):
    self._iterator = iterator
    self._queue: queue.Queue = queue.Queue(maxsize)
    self._sentinel = object()
    self._stop_event = threading.Event()
    self._error = None
    self._worker = threading.Thread(
        target=self._run, name='prefetch-data', daemon=True)
    self._worker.start()

  def _run(self):
    try:
      while not self._stop_event.is_set():
        item = next(self._iterator)
        self._queue.put(item)
    except StopIteration:
      self._stop_event.set()
      self._queue.put(self._sentinel)
    except Exception as exc:
      self._stop_event.set()
      self._error = exc
      self._queue.put(self._sentinel)

  def __iter__(self):
    return self

  def __next__(self):
    item = self._queue.get()
    if item is self._sentinel:
      if self._error is not None:
        raise self._error
      raise StopIteration
    return item

  def __getattr__(self, name):
    return getattr(self._iterator, name)

  def close(self):
    self._stop_event.set()
    try:
      self._queue.put_nowait(self._sentinel)
    except queue.Full:
      pass
    if hasattr(self._iterator, 'close'):
      try:
        self._iterator.close()
      except Exception:
        pass
    if self._worker.is_alive():
      self._worker.join(timeout=0.1)

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
