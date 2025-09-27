import atexit
import collections
import dataclasses
import itertools
import logging
import json
import multiprocessing as mp
import os
import random
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
  swap: bool = True  # yield swapped versions of each replay
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

        meta_slots = [empty_player, empty_player, empty_player, empty_player]
        meta_slots[main_index] = player
        meta_slots[opponent_order[0]] = opponent
        meta_slots[opponent_order[1]] = empty_player

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
            opponent_order=opponent_order,
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
        replays.append(ReplayInfo(
            replay_path,
            player_index,
            teammate_index,
            players[player_index].name,
            replay_meta,
            other_ports,
        ))
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
      name_map: Optional[dict[str, int]] = None,
  ):
    self.replays = replays
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.damage_ratio = damage_ratio
    self.compressed = compressed
    self.batch_counter = 0
    self.balance_characters = balance_characters

    self.replay_counter = 0
    replays = self.iter_replays()
    self.managers = [
        TrajectoryManager(
            replays,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            compressed=compressed,
            game_filter=self.is_allowed)
        for _ in range(batch_size)]

    self.allowed_characters = _charset(allowed_characters)
    self.allowed_opponents = _charset(allowed_opponents)
    self.name_map = name_map or {}
    self.encode_name = nametags.name_encoder(self.name_map)

  def iter_replays(self) -> Iterator[ReplayInfo]:
    replay_iter = itertools.cycle(self.replays)

    if not self.balance_characters:
      for replay in replay_iter:
        self.replay_counter += 1
        yield replay
      return

    by_character: dict[int, list[ReplayInfo]] = collections.defaultdict(list)
    skipped = 0

    for replay in self.replays:
      try:
        character = int(replay.main_player.character)
      except Exception:
        skipped += 1
        continue

      by_character[character].append(replay)

    if len(by_character) <= 1:
      if skipped:
        logging.debug(
            'Skipping character balancing because %d replays lacked metadata.',
            skipped)
      for replay in replay_iter:
        self.replay_counter += 1
        yield replay
      return

    character_counts: dict[str, int] = {}
    for char, entries in by_character.items():
      try:
        char_enum = melee.Character(char)
        char_name = char_enum.name.lower()
      except ValueError:
        char_name = str(char)
      character_counts[char_name] = len(entries)

    if skipped:
      logging.info(
          'Skipped %d replay(s) without character metadata while balancing.',
          skipped)
    logging.info('Character balance counts: %s', character_counts)

    iterators = [itertools.cycle(entries) for entries in by_character.values()]
    balanced_iter = utils.interleave(*iterators)
    combined_iter = utils.interleave(balanced_iter, replay_iter)

    for replay in combined_iter:
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
    batch: Batch = self.process_batch(
        [m.grab_chunk() for m in self.managers])
    epoch = self.replay_counter / len(self.replays)
    self.batch_counter += 1
    assert batch.frames.state_action.state.stage.shape[-1] == self.chunk_size
    assert batch.frames.reward.shape[-1] == self.chunk_size - 1
    return batch, epoch

def produce_batches(data_source_kwargs, batch_queue):
  data_source = DataSource(**data_source_kwargs)
  while True:
    batch_queue.put(next(data_source))

class DataSourceMP:
  def __init__(self, buffer=4, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    self.batch_queue = mp.Queue(buffer)
    self.process = mp.Process(
        target=produce_batches, args=(kwargs, self.batch_queue))
    self.process.start()

    atexit.register(self.batch_queue.close)
    atexit.register(self.process.terminate)

  def __next__(self) -> Tuple[Batch, float]:
    return self.batch_queue.get()

  def __del__(self):
    self.process.terminate()

@dataclasses.dataclass
class DataConfig:
  batch_size: int = 32
  unroll_length: int = 64
  damage_ratio: float = 0.01
  compressed: bool = True
  in_parallel: bool = True
  balance_characters: bool = False

def make_source(
    in_parallel: bool,
    **kwargs):
  constructor = DataSourceMP if in_parallel else DataSource
  return constructor(**kwargs)

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
