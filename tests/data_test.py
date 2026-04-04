import contextlib
import io
import json
import pathlib
import shutil
import sys
import tempfile
import types as pytypes
import unittest

import numpy as np

from slippi_ai import data, types, paths
from slippi_db import file_layout

import peppi_py

if 'py7zr' not in sys.modules:  # pragma: no cover
  stub_module = pytypes.ModuleType('py7zr')

  class _SevenZipStub:  # pragma: no cover
    def __init__(self, *args, **kwargs):
      raise ImportError('py7zr is required for archive handling')

    def getnames(self):
      raise ImportError('py7zr is required for archive handling')

    def extract(self, *args, **kwargs):
      raise ImportError('py7zr is required for archive handling')

  stub_module.SevenZipFile = _SevenZipStub
  sys.modules['py7zr'] = stub_module

from slippi_db import parse_local
from slippi_db import parse_peppi
from slippi_db import utils
from slippi_db.scripts import make_local_dataset


_SAMPLE_FRAMES = (0, 60, 600, -1)


def _summarize_game(game_array: types.GAME_TYPE) -> dict:
  game_nt = types.game_array_to_nt(game_array)
  frame_count = len(game_nt.stage)

  summary = {
      'frame_count': int(frame_count),
      'stage': int(game_nt.stage[0]),
      'is_teams': bool(game_nt.is_teams[0]),
      'samples': {},
  }

  for frame_idx in _SAMPLE_FRAMES:
    idx = frame_idx if frame_idx >= 0 else frame_count + frame_idx
    if idx < 0 or idx >= frame_count:
      continue

    players_snapshot = {}
    for port in range(4):
      player = getattr(game_nt, f'p{port}')
      players_snapshot[f'p{port}'] = {
          'x': round(float(player.x[idx]), 4),
          'y': round(float(player.y[idx]), 4),
          'percent': round(float(player.percent[idx]), 2),
          'action': int(player.action[idx]),
          'stocks': int(player.stocks_left[idx]),
          'is_dead': bool(player.is_dead[idx]),
      }

    summary['samples'][str(frame_idx)] = {
        'randall_phase': int(game_nt.randall_phase[idx]),
        'players': players_snapshot,
    }

  return summary


def _make_controller():
  zeros_bool = np.zeros(1, dtype=np.bool_)
  buttons = types.Buttons(
      A=zeros_bool.copy(),
      B=zeros_bool.copy(),
      X=zeros_bool.copy(),
      Y=zeros_bool.copy(),
      Z=zeros_bool.copy(),
      L=zeros_bool.copy(),
      R=zeros_bool.copy(),
      D_UP=zeros_bool.copy(),
  )
  zeros_float = np.zeros(1, dtype=np.float32)
  stick = types.Stick(x=zeros_float.copy(), y=zeros_float.copy())
  return types.Controller(
      main_stick=stick,
      c_stick=stick,
      shoulder=zeros_float.copy(),
      buttons=buttons,
  )


def _make_nana(dead: bool = True) -> types.Nana:
  zeros_bool = np.zeros(1, dtype=np.bool_)
  zeros_float = np.zeros(1, dtype=np.float32)
  zeros_uint16 = np.zeros(1, dtype=np.uint16)
  zeros_uint8 = np.zeros(1, dtype=np.uint8)

  return types.Nana(
      exists=zeros_bool.copy(),
      percent=zeros_uint16.copy(),
      facing=zeros_bool.copy(),
      x=zeros_float.copy(),
      y=zeros_float.copy(),
      action=zeros_uint16.copy(),
      invulnerable=zeros_bool.copy(),
      character=zeros_uint8.copy(),
      jumps_left=zeros_uint8.copy(),
      shield_strength=zeros_float.copy(),
      on_ground=np.logical_not(zeros_bool.copy()) if dead else np.ones(1, dtype=np.bool_),
  )


def _make_item() -> types.Item:
  zeros_bool = np.zeros(1, dtype=np.bool_)
  zeros_float = np.zeros(1, dtype=np.float32)
  zeros_uint16 = np.zeros(1, dtype=np.uint16)
  zeros_uint8 = np.zeros(1, dtype=np.uint8)
  return types.Item(
      exists=zeros_bool.copy(),
      type=zeros_uint16.copy(),
      state=zeros_uint8.copy(),
      x=zeros_float.copy(),
      y=zeros_float.copy(),
  )


def _make_items() -> types.Items:
  return types.Items(**{f'item_{i}': _make_item() for i in range(types.MAX_ITEMS)})


def _make_player(character: int, dead: bool = False) -> types.Player:
  zeros_float = np.zeros(1, dtype=np.float32)
  zeros_uint16 = np.zeros(1, dtype=np.uint16)
  zeros_uint8 = np.zeros(1, dtype=np.uint8)
  bool_val = np.full(1, dead, dtype=np.bool_)

  return types.Player(
      percent=zeros_uint16.copy(),
      facing=np.ones(1, dtype=np.bool_),
      x=zeros_float.copy(),
      y=zeros_float.copy(),
      action=zeros_uint16.copy(),
      invulnerable=np.zeros(1, dtype=np.bool_),
      character=np.full(1, character, dtype=np.uint8),
      jumps_left=zeros_uint8.copy(),
      shield_strength=zeros_float.copy(),
      on_ground=np.ones(1, dtype=np.bool_),
      is_dead=bool_val,
      stocks_left=np.full(1, 4 if not dead else 0, dtype=np.uint8),
      controller=_make_controller(),
      nana=_make_nana(dead=True),
  )


def _make_game() -> types.Game:
  stage = np.zeros(1, dtype=np.uint8)
  randall_phase = np.zeros(1, dtype=np.float32)
  randall = types.Randall(
      x=np.zeros(1, dtype=np.float32),
      y=np.zeros(1, dtype=np.float32),
  )
  items = _make_items()
  is_teams = np.zeros(1, dtype=np.bool_)

  p0 = _make_player(1)
  opponent = _make_player(22)
  empty = _make_player(0, dead=True)

  return types.Game(
      p0=p0,
      p1=empty,
      p2=opponent,
      p3=empty,
      stage=stage,
      randall_phase=randall_phase,
      randall=randall,
      items=items,
      is_teams=is_teams,
  )


def _make_replay_info(character: int, slug: str) -> data.ReplayInfo:
  player = data.PlayerMeta(character=character, name=f'player{character}', team=0)
  empty = data.PlayerMeta(character=0, name='', team=1)
  meta = data.ReplayMeta(
      p0=player,
      p1=empty,
      p2=empty,
      p3=empty,
      stage=0,
      slp_md5=f'slp_{slug}',
      is_singles=True,
  )

  return data.ReplayInfo(
      path=f'/tmp/{slug}.parquet',
      main_player_index=0,
      teammate_index=1,
      main_player_name=player.name,
      meta=meta,
      opponent_order=(2, 3),
  )


def _make_doubles_replay_info(character: int, slug: str) -> data.ReplayInfo:
  player = data.PlayerMeta(character=character, name=f'd_player{character}', team=0)
  teammate = data.PlayerMeta(character=character + 100, name=f'd_team{character}', team=0)
  opponent_a = data.PlayerMeta(character=character + 200, name=f'd_opp{character}', team=1)
  opponent_b = data.PlayerMeta(character=character + 201, name=f'd_opp{character}b', team=1)
  meta = data.ReplayMeta(
      p0=player,
      p1=teammate,
      p2=opponent_a,
      p3=opponent_b,
      stage=1,
      slp_md5=f'slp_{slug}',
      is_singles=False,
  )

  return data.ReplayInfo(
      path=f'/tmp/{slug}.parquet',
      main_player_index=0,
      teammate_index=1,
      main_player_name=player.name,
      meta=meta,
      opponent_order=(2, 3),
  )


class DataTest(unittest.TestCase):

  def test_replays_from_meta_singles(self):
    players_meta = [
        dict(port=0, character=1, type=0, name_tag='', netplay=dict(name='', code='', suid=''), team=0),
        dict(port=1, character=22, type=0, name_tag='', netplay=dict(name='', code='', suid=''), team=1),
    ]
    meta_row = dict(
        name='singles.slp',
        slp_md5='cafebabe',  # ends with even hex digit -> opponent_order stays sorted
        slp_size=0,
        lastFrame=100,
        slippi_version=[3, 14, 0],
        num_players=2,
        players=players_meta,
        stage=2,
        timer=480,
        is_teams=False,
        winner=0,
        valid=True,
        is_training=True,
        not_training_reason='',
        pq_size=0,
        raw='ignored.zip',
        compression='zlib',
    )

    with tempfile.TemporaryDirectory() as tmp:
      data_dir = pathlib.Path(tmp) / 'games'
      data_dir.mkdir()
      (data_dir / meta_row['slp_md5']).touch()

      meta_path = pathlib.Path(tmp) / 'meta.json'
      meta_path.write_text(json.dumps([meta_row]))

      cfg = data.DatasetConfig(
          data_dir=str(data_dir),
          meta_path=str(meta_path),
          allowed_characters='all',
          allowed_opponents='all',
          swap=False,
      )

      replays = data.replays_from_meta(cfg)

    self.assertEqual(len(replays), 2)
    replays_by_main = {info.main_player_index: info for info in replays}

    first = replays_by_main[0]
    self.assertEqual(first.teammate_index, 1)
    self.assertEqual(first.opponent_order, (2, 3))
    self.assertEqual(first.meta.p2.character, 22)
    self.assertEqual(first.meta.p3.character, 0)

    second = replays_by_main[2]
    self.assertEqual(second.teammate_index, 1)
    self.assertEqual(second.opponent_order, (0, 3))
    self.assertEqual(second.meta.p0.character, 1)
    self.assertEqual(second.meta.p2.character, 22)
    self.assertEqual(second.meta.p3.character, 0)

  def test_replays_from_meta_doubles(self):
    players_meta = [
        dict(port=i, character=i + 1, type=0, name_tag='',
             netplay=dict(name='', code='', suid=''), team=0 if i < 2 else 1)
        for i in range(4)
    ]
    meta_row = dict(
        name='doubles.slp',
        slp_md5='facefeed',
        slp_size=0,
        lastFrame=200,
        slippi_version=[3, 14, 0],
        num_players=4,
        players=players_meta,
        stage=3,
        timer=480,
        is_teams=True,
        winner=0,
        valid=True,
        is_training=True,
        not_training_reason='',
        pq_size=0,
        raw='ignored.zip',
        compression='zlib',
    )

    with tempfile.TemporaryDirectory() as tmp:
      data_dir = pathlib.Path(tmp) / 'games'
      data_dir.mkdir()
      (data_dir / meta_row['slp_md5']).touch()

      meta_path = pathlib.Path(tmp) / 'meta.json'
      meta_path.write_text(json.dumps([meta_row]))

      cfg = data.DatasetConfig(
          data_dir=str(data_dir),
          meta_path=str(meta_path),
          allowed_characters='all',
          allowed_opponents='all',
          swap=False,
      )

      replays = data.replays_from_meta(cfg)

    self.assertEqual(len(replays), 4)
    for info in replays:
      # teammate should share team id
      team_ids = [info.meta.p0.team, info.meta.p1.team, info.meta.p2.team, info.meta.p3.team]
      self.assertEqual(team_ids[info.main_player_index], team_ids[info.teammate_index])

      expected_opponents = tuple(i for i in range(4)
                                 if i not in (info.main_player_index, info.teammate_index))
      self.assertEqual(info.opponent_order, expected_opponents)

  def test_replays_from_meta_respects_allowed_main_player_indices(self):
    players_meta = [
        dict(port=0, character=1, type=0, name_tag='',
             netplay=dict(name='Enzyme', code='ENZYME#0', suid=''), team=0),
        dict(port=1, character=22, type=0, name_tag='',
             netplay=dict(name='Tempo', code='TEMP#0', suid=''), team=0),
        dict(port=2, character=9, type=0, name_tag='',
             netplay=dict(name='OpponentA', code='OPPA#1', suid=''), team=1),
        dict(port=3, character=18, type=0, name_tag='',
             netplay=dict(name='OpponentB', code='OPPB#1', suid=''), team=1),
    ]
    meta_row = dict(
        name='doubles_filtered.slp',
        slp_md5='facefeed',
        slp_size=0,
        lastFrame=100,
        slippi_version=[3, 14, 0],
        num_players=4,
        players=players_meta,
        stage=2,
        timer=480,
        is_teams=True,
        winner=0,
        valid=True,
        is_training=True,
        not_training_reason='',
        pq_size=0,
        raw='ignored.zip',
        compression='zlib',
        allowed_main_player_indices=[1],
    )

    with tempfile.TemporaryDirectory() as tmp:
      data_dir = pathlib.Path(tmp) / 'games'
      data_dir.mkdir()
      (data_dir / meta_row['slp_md5']).touch()

      meta_path = pathlib.Path(tmp) / 'meta.json'
      meta_path.write_text(json.dumps([meta_row]))

      cfg = data.DatasetConfig(
          data_dir=str(data_dir),
          meta_path=str(meta_path),
          allowed_characters='all',
          allowed_opponents='all',
          swap=False,
      )

      replays = data.replays_from_meta(cfg)

    self.assertEqual(len(replays), 1)
    self.assertEqual(replays[0].main_player_index, 1)
    self.assertEqual(replays[0].main_player_name, 'TEMP#0')

  def test_replays_from_meta_prefers_hashed_layout(self):
    md5 = 'abcdef0123456789abcdef0123456789'
    players_meta = [
        dict(port=0, character=1, type=0, name_tag='', netplay=dict(name='', code='', suid=''), team=0),
        dict(port=1, character=2, type=0, name_tag='', netplay=dict(name='', code='', suid=''), team=1),
    ]
    meta_row = dict(
        name='singles.slp',
        slp_md5=md5,
        slp_size=0,
        lastFrame=100,
        slippi_version=[3, 14, 0],
        num_players=2,
        players=players_meta,
        stage=2,
        timer=480,
        is_teams=False,
        winner=0,
        valid=True,
        is_training=True,
        not_training_reason='',
        pq_size=0,
        raw='ignored.zip',
        compression='zlib',
    )

    with tempfile.TemporaryDirectory() as tmp:
      data_dir = pathlib.Path(tmp) / 'games'
      hashed_dir = data_dir / md5[:file_layout.PARQUET_PREFIX_LEN]
      hashed_dir.mkdir(parents=True)
      (hashed_dir / md5).touch()

      meta_path = pathlib.Path(tmp) / 'meta.json'
      meta_path.write_text(json.dumps([meta_row]))

      cfg = data.DatasetConfig(
          data_dir=str(data_dir),
          meta_path=str(meta_path),
          allowed_characters='all',
          allowed_opponents='all',
      )

      replays = data.replays_from_meta(cfg)

    self.assertEqual(len(replays), 2)
    for replay in replays:
      self.assertEqual(replay.path, str(hashed_dir / md5))

  def test_end_to_end_parse_pipeline(self):
    singles_path = pathlib.Path(__file__).parent / 'data' / 'replays' / 'test_singles_game.slp'
    doubles_path = pathlib.Path(__file__).parent / 'data' / 'replays' / 'test_doubles_game.slp'

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      raw_dir = root / 'Raw'
      parsed_dir = root / 'Parsed'
      raw_dir.mkdir()
      parsed_dir.mkdir()

      parsed_entries = []
      for source in [singles_path, doubles_path]:
        shutil.copy(source, raw_dir / source.name)
        local_file = utils.SimplePath(str(raw_dir), source.name)
        result = parse_local.parse_slp(local_file, str(parsed_dir), tmp)
        self.assertTrue(result['valid'], msg=f"Parse failed for {source.name}: {result}")
        self.assertTrue(result['is_training'])
        result.setdefault('raw', source.name)
        parsed_entries.append(result)

      parse_local.save_slp_meta(parsed_entries, str(root / 'parsed.pkl'))

      make_local_dataset.build_meta(
          root,
          doubles_only=False,
          winner_only=False,
          make_tar=False,
          allowed_players=None,
          quiet=True,
      )

      meta_path = root / 'meta.json'
      cfg = data.DatasetConfig(
          data_dir=str(parsed_dir),
          meta_path=str(meta_path),
          allowed_characters='all',
          allowed_opponents='all',
          test_ratio=0.0,
      )

      train, test = data.train_test_split(cfg)
      self.assertFalse(test)
      self.assertEqual(len(train), 6)

      singles_infos = [info for info in train if info.meta.is_singles]
      doubles_infos = [info for info in train if not info.meta.is_singles]

      self.assertEqual(len(singles_infos), 2)
      singles_by_main = {info.main_player_index: info for info in singles_infos}
      self.assertSetEqual(set(singles_by_main.keys()), {0, 2})

      md5_singles = pathlib.Path(singles_path).read_bytes()
      singles_md5 = utils.md5(md5_singles)
      self.assertEqual(singles_by_main[0].meta.slp_md5, singles_md5)
      self.assertEqual(singles_by_main[2].meta.slp_md5, singles_md5)
      self.assertEqual(singles_by_main[0].opponent_order, (2, 3))
      self.assertEqual(singles_by_main[2].opponent_order, (0, 3))

      self.assertEqual(len(doubles_infos), 4)
      doubles_by_main = {info.main_player_index: info for info in doubles_infos}
      self.assertSetEqual(set(doubles_by_main.keys()), {0, 1, 2, 3})

      md5_doubles = utils.md5(pathlib.Path(doubles_path).read_bytes())
      for info in doubles_infos:
        self.assertEqual(info.meta.slp_md5, md5_doubles)
        expected_opponents = tuple(i for i in range(4)
                                   if i not in (info.main_player_index, info.teammate_index))
        self.assertEqual(info.opponent_order, expected_opponents)

  def test_build_meta_allowed_players_file_annotates_and_summarizes(self):
    kept_row = dict(
        name='kept.slp',
        slp_md5='facefeed',
        slp_size=0,
        lastFrame=100,
        slippi_version=[3, 14, 0],
        num_players=4,
        players=[
            dict(port=0, character=1, type=0, name_tag='',
                 netplay=dict(name='Enzyme', code='ENZYME#0', suid=''), team=0),
            dict(port=1, character=22, type=0, name_tag='',
                 netplay=dict(name='Tempo', code='TEMP#0', suid=''), team=0),
            dict(port=2, character=9, type=0, name_tag='',
                 netplay=dict(name='OpponentA', code='OPPA#1', suid=''), team=1),
            dict(port=3, character=18, type=0, name_tag='',
                 netplay=dict(name='OpponentB', code='OPPB#1', suid=''), team=1),
        ],
        stage=2,
        timer=480,
        is_teams=True,
        winner=0,
        valid=True,
        is_training=True,
        not_training_reason='',
        pq_size=0,
        raw='ignored.zip',
        compression='zlib',
    )
    dropped_row = dict(
        kept_row,
        name='dropped.slp',
        slp_md5='deadc0de',
        players=[
            dict(port=0, character=3, type=0, name_tag='',
                 netplay=dict(name='Someone', code='SOME#1', suid=''), team=0),
            dict(port=1, character=4, type=0, name_tag='',
                 netplay=dict(name='Else', code='ELSE#1', suid=''), team=0),
            dict(port=2, character=9, type=0, name_tag='',
                 netplay=dict(name='OpponentA', code='OPPA#1', suid=''), team=1),
            dict(port=3, character=18, type=0, name_tag='',
                 netplay=dict(name='OpponentB', code='OPPB#1', suid=''), team=1),
        ],
    )

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      parsed_dir = root / 'Parsed'
      parsed_dir.mkdir()

      for md5 in (kept_row['slp_md5'], dropped_row['slp_md5']):
        hashed_dir = parsed_dir / md5[:file_layout.PARQUET_PREFIX_LEN]
        hashed_dir.mkdir(parents=True, exist_ok=True)
        (hashed_dir / md5).touch()

      with open(root / 'parsed.pkl', 'wb') as f:
        import pickle
        pickle.dump([kept_row, dropped_row], f)

      whitelist_path = root / 'players.txt'
      whitelist_path.write_text('ENZYME#0\nTempo\nMissingPlayer\n')

      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout):
        rows = make_local_dataset.build_meta(
            root,
            doubles_only=True,
            winner_only=False,
            make_tar=False,
            allowed_players_file=str(whitelist_path),
            quiet=False,
        )

      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]['allowed_main_player_indices'], [0, 1])

      with open(root / 'meta.json') as f:
        meta_rows = json.load(f)
      self.assertEqual(len(meta_rows), 1)
      self.assertEqual(meta_rows[0]['allowed_main_player_indices'], [0, 1])

      output = stdout.getvalue()
      self.assertIn('Selected player replay counts:', output)
      self.assertIn('Enzyme: 1', output)
      self.assertIn('Tempo: 1', output)
      self.assertIn('MissingPlayer: 0', output)
      self.assertNotIn('OpponentA', output)
      self.assertNotIn('OpponentB', output)
      self.assertNotIn('Someone', output)
      self.assertNotIn('Else', output)

  def test_train_test_split_is_deterministic(self):
    cfg = data.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        test_ratio=0.5,
        seed=123,
    )

    train_a, test_a = data.train_test_split(cfg)
    train_b, test_b = data.train_test_split(cfg)

    self.assertEqual(train_a, train_b)
    self.assertEqual(test_a, test_b)

  def _meta_from_slp(self, source: pathlib.Path, parse_result: dict) -> dict:
    game = peppi_py.read_slippi(str(source))
    start = game.start if hasattr(game, 'start') else game['start']
    players_raw = start.get('players', [])

    players = []
    for idx, player in enumerate(players_raw):
      port = player.get('port', idx)
      if isinstance(port, str) and port.upper().startswith('P'):
        try:
          port_num = int(port[1:]) - 1
        except ValueError:
          port_num = idx
      else:
        port_num = int(port)

      netplay = player.get('netplay', {})
      team = player.get('team')
      if team is None:
        team = 0 if port_num in (0, 1) else 1

      players.append(dict(
          port=port_num,
          character=player.get('character', 0),
          type=0,
          name_tag=player.get('name_tag', ''),
          netplay=dict(
              name=netplay.get('name', ''),
              code=netplay.get('code', ''),
              suid=netplay.get('suid', ''),
          ),
          team=team,
      ))

    players.sort(key=lambda p: p['port'])

    is_teams = start.get('is_teams')
    if is_teams is None:
      is_teams = len(players) == 4
    stage = start.get('stage', 0)

    return dict(
        name=source.name,
        raw=source.name,
        slp_md5=parse_result['slp_md5'],
        slp_size=parse_result.get('slp_size', 0),
        stage=stage,
        num_players=len(players),
        is_training=True,
        is_teams=is_teams,
        valid=True,
        players=players,
    )

  def _build_replay_info(self, *, main_index: int, slp_md5: str,
                         player_char: int, opponent_char: int) -> data.ReplayInfo:
    empty_meta = data.PlayerMeta(character=0, name='', team=0)
    player_meta = data.PlayerMeta(character=player_char, name='Self', team=0)
    opponent_meta = data.PlayerMeta(character=opponent_char, name='Opponent', team=0)

    meta_slots = [empty_meta, empty_meta, empty_meta, empty_meta]
    meta_slots[main_index] = player_meta

    teammate_index = 1
    other_ports = [i for i in range(4) if i not in (main_index, teammate_index)]
    hash_val = int(slp_md5[-1], 16)
    opponent_order = tuple(other_ports if hash_val % 2 == 0 else reversed(other_ports))

    opponent_port, empty_port = opponent_order
    meta_slots[opponent_port] = opponent_meta
    meta_slots[empty_port] = empty_meta

    meta = data.ReplayMeta(
        p0=meta_slots[0],
        p1=meta_slots[1],
        p2=meta_slots[2],
        p3=meta_slots[3],
        stage=8,
        slp_md5=slp_md5,
        is_singles=True,
    )

    return data.ReplayInfo(
        path='ignored',
        main_player_index=main_index,
        teammate_index=teammate_index,
        main_player_name='Self',
        meta=meta,
        opponent_order=opponent_order,
    )

  def test_singles_first_perspective_layout(self):
    game = _make_game()
    info_even = self._build_replay_info(
        main_index=0,
        slp_md5='deadbeee',
        player_char=1,
        opponent_char=22,
    )
    swapped_even = data.swap_players(game, info_even)

    self.assertEqual(int(swapped_even.p0.character[0]), 1)
    self.assertTrue(swapped_even.p1.is_dead[0])
    self.assertEqual(int(swapped_even.p2.character[0]), 22)
    self.assertTrue(swapped_even.p3.is_dead[0])

    info_odd = self._build_replay_info(
        main_index=0,
        slp_md5='deadbeef1',
        player_char=1,
        opponent_char=22,
    )
    swapped_odd = data.swap_players(game, info_odd)
    self.assertEqual(int(swapped_odd.p0.character[0]), 1)
    self.assertTrue(swapped_odd.p1.is_dead[0])
    self.assertTrue(swapped_odd.p2.is_dead[0])
    self.assertEqual(int(swapped_odd.p3.character[0]), 22)

  def test_parsed_replays_match_golden(self):
    golden_path = pathlib.Path(__file__).parent / 'golden' / 'parse_peppi_summary.json'
    with golden_path.open('r', encoding='utf-8') as fp:
      golden = json.load(fp)

    replay_dir = pathlib.Path(__file__).parent / 'data' / 'replays'

    for relative_path, expected in golden.items():
      replay_path = replay_dir / relative_path
      game_array = parse_peppi.get_slp(str(replay_path))
      actual = _summarize_game(game_array)
      self.assertDictEqual(
          actual,
          expected,
          msg=f'{relative_path} parsing no longer matches golden summary',
      )

  def test_singles_second_perspective_layout(self):
    game = _make_game()

    info_even = self._build_replay_info(
        main_index=2,
        slp_md5='deadbeee',
        player_char=22,
        opponent_char=1,
    )
    swapped_even = data.swap_players(game, info_even)
    self.assertEqual(int(swapped_even.p0.character[0]), 22)
    self.assertTrue(swapped_even.p1.is_dead[0])
    self.assertEqual(int(swapped_even.p2.character[0]), 1)
    self.assertTrue(swapped_even.p3.is_dead[0])

    info_odd = self._build_replay_info(
        main_index=2,
        slp_md5='deadbeef1',
        player_char=22,
        opponent_char=1,
    )
    swapped_odd = data.swap_players(game, info_odd)
    self.assertEqual(int(swapped_odd.p0.character[0]), 22)
    self.assertTrue(swapped_odd.p1.is_dead[0])
    self.assertTrue(swapped_odd.p2.is_dead[0])
    self.assertEqual(int(swapped_odd.p3.character[0]), 1)

  def test_doubles_swap_layout(self):
    zeros_float = np.zeros(1, dtype=np.float32)
    zeros_uint16 = np.zeros(1, dtype=np.uint16)
    zeros_uint8 = np.zeros(1, dtype=np.uint8)
    controller = _make_controller()

    def make_full_player(char: int) -> types.Player:
      return types.Player(
          percent=zeros_uint16.copy(),
          facing=np.ones(1, dtype=np.bool_),
          x=zeros_float.copy(),
          y=zeros_float.copy(),
          action=zeros_uint16.copy(),
          invulnerable=np.zeros(1, dtype=np.bool_),
          character=np.full(1, char, dtype=np.uint8),
          jumps_left=zeros_uint8.copy(),
          shield_strength=zeros_float.copy(),
          on_ground=np.ones(1, dtype=np.bool_),
          is_dead=np.full(1, False, dtype=np.bool_),
          stocks_left=np.full(1, 4, dtype=np.uint8),
          controller=controller,
          nana=_make_nana(dead=False),
      )

    game = types.Game(
        p0=make_full_player(1),
        p1=make_full_player(2),
        p2=make_full_player(3),
        p3=make_full_player(4),
        stage=np.zeros(1, dtype=np.uint8),
        randall_phase=np.zeros(1, dtype=np.float32),
        randall=types.Randall(
            x=np.zeros(1, dtype=np.float32),
            y=np.zeros(1, dtype=np.float32),
        ),
        items=_make_items(),
        is_teams=np.ones(1, dtype=np.bool_),
    )

    meta = data.ReplayMeta(
        p0=data.PlayerMeta(character=1, name='A', team=0),
        p1=data.PlayerMeta(character=2, name='B', team=0),
        p2=data.PlayerMeta(character=3, name='C', team=1),
        p3=data.PlayerMeta(character=4, name='D', team=1),
        stage=2,
        slp_md5='beefdead',
        is_singles=False,
    )

    info_self = data.ReplayInfo(
        path='ignored',
        main_player_index=0,
        teammate_index=1,
        main_player_name='A',
        meta=meta,
        opponent_order=(2, 3),
    )
    swapped_self = data.swap_players(game, info_self)
    self.assertEqual(int(swapped_self.p0.character[0]), 1)
    self.assertEqual(int(swapped_self.p1.character[0]), 2)
    self.assertEqual(int(swapped_self.p2.character[0]), 3)
    self.assertEqual(int(swapped_self.p3.character[0]), 4)

    info_opponent = data.ReplayInfo(
        path='ignored',
        main_player_index=2,
        teammate_index=3,
        main_player_name='C',
        meta=meta,
        opponent_order=(0, 1),
    )
    swapped_opponent = data.swap_players(game, info_opponent)
    self.assertEqual(int(swapped_opponent.p0.character[0]), 3)
    self.assertEqual(int(swapped_opponent.p1.character[0]), 4)
    self.assertEqual(int(swapped_opponent.p2.character[0]), 1)
    self.assertEqual(int(swapped_opponent.p3.character[0]), 2)

  def test_iter_replays_balances_characters(self):
    replays = [
        _make_replay_info(1, 'fox'),
        _make_replay_info(2, 'falco'),
        _make_replay_info(3, 'sheik'),
    ]

    data_source = data.DataSource.__new__(data.DataSource)
    data_source.replays = replays
    data_source.balance_characters = True
    data_source.balance_singles_doubles = False
    data_source.character_balance_ratio = 0.5
    data_source.replay_counter = 0

    iterator = data_source.iter_replays()
    observed = {next(iterator).main_player.character for _ in range(2 * len(replays))}
    self.assertEqual(observed, {1, 2, 3})

  def test_iter_replays_balance_falls_back_with_single_character(self):
    replays = [
        _make_replay_info(4, 'marth_a'),
        _make_replay_info(4, 'marth_b'),
    ]

    data_source = data.DataSource.__new__(data.DataSource)
    data_source.replays = replays
    data_source.balance_characters = True
    data_source.balance_singles_doubles = False
    data_source.character_balance_ratio = 0.5
    data_source.replay_counter = 0

    iterator = data_source.iter_replays()
    observed_paths = [next(iterator).path for _ in range(4)]
    self.assertEqual(observed_paths, [replays[0].path, replays[1].path] * 2)
    self.assertEqual(data_source.replay_counter, 4)

  def test_iter_replays_includes_replays_without_metadata(self):
    valid_replays = [
        _make_replay_info(5, 'falcon'),
        _make_replay_info(6, 'ganon'),
    ]
    missing_meta = data.ReplayInfo(
        path='/tmp/invalid.parquet',
        main_player_index=0,
        teammate_index=1,
        main_player_name='unknown',
        meta=(),
        opponent_order=(2, 3),
    )

    replays = valid_replays + [missing_meta]

    data_source = data.DataSource.__new__(data.DataSource)
    data_source.replays = replays
    data_source.balance_characters = True
    data_source.balance_singles_doubles = False
    data_source.character_balance_ratio = 0.5
    data_source.replay_counter = 0

    iterator = data_source.iter_replays()
    observed_paths = [next(iterator).path for _ in range(6)]
    self.assertIn('/tmp/invalid.parquet', observed_paths)

  def test_replay_stream_enforces_mode_balance(self):
    singles = [
        _make_replay_info(1, 'singles_mode_a'),
        _make_replay_info(2, 'singles_mode_b'),
    ]
    doubles = [
        _make_doubles_replay_info(10 + i, f'doubles_mode_{i}')
        for i in range(4)
    ]

    iterator = data.replay_stream(
        singles + doubles,
        balance_characters=False,
        balance_singles_doubles=True,
    )

    samples = [next(iterator) for _ in range(40)]
    singles_count = sum(1 for info in samples if info.meta.is_singles)
    doubles_count = len(samples) - singles_count
    self.assertEqual(singles_count, doubles_count)

  def test_character_balance_ratio_scales_balanced_weight(self):
    dominant = [_make_replay_info(1, f'dominant_{i}') for i in range(5)]
    rare = [_make_replay_info(2, 'rare')]
    replays = dominant + rare

    sample_size = 120

    def _ratio(r):
      iterator = data.replay_stream(
          replays,
          balance_characters=True,
          balance_singles_doubles=False,
          character_balance_ratio=r,
      )
      samples = [next(iterator) for _ in range(sample_size)]
      rare_count = sum(1 for info in samples if info.main_player.character == 2)
      return rare_count / sample_size

    raw_ratio = _ratio(0.0)
    partial_ratio = _ratio(0.2)
    half_ratio = _ratio(0.5)

    self.assertGreater(partial_ratio, raw_ratio)
    self.assertGreater(half_ratio, partial_ratio)

  def test_iter_replays_balances_modes_and_characters(self):
    singles = [
        _make_replay_info(1, 'singles_a'),
        _make_replay_info(2, 'singles_b'),
    ]
    doubles = [
        _make_doubles_replay_info(10, 'doubles_a'),
        _make_doubles_replay_info(11, 'doubles_b'),
        _make_doubles_replay_info(12, 'doubles_c'),
    ]
    replays = singles + doubles

    data_source = data.DataSource.__new__(data.DataSource)
    data_source.replays = replays
    data_source.balance_characters = True
    data_source.balance_singles_doubles = True
    data_source.character_balance_ratio = 0.5
    data_source.replay_counter = 0

    iterator = data_source.iter_replays()
    samples = [next(iterator) for _ in range(12)]

    singles_count = sum(1 for info in samples if info.meta.is_singles)
    doubles_count = len(samples) - singles_count
    self.assertEqual(singles_count, doubles_count)

    singles_chars = {info.main_player.character for info in samples if info.meta.is_singles}
    self.assertSetEqual(singles_chars, {1, 2})

    doubles_chars = {info.main_player.character for info in samples if not info.meta.is_singles}
    self.assertSetEqual(doubles_chars, {10, 11, 12})


if __name__ == '__main__':
  unittest.main(failfast=True)
