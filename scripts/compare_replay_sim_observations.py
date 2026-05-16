import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import melee
import numpy as np

from slippi_ai import dolphin
from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import tf_checkpoint
from slippi_db import parse_libmelee


OPENING_ALIGNMENT_TOLERANCE = 1e-4


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--replay', required=True)
  parser.add_argument('--model-path', required=True)
  parser.add_argument('--frames', type=int, default=3000)
  parser.add_argument('--start-frame', type=int, default=0)
  parser.add_argument('--sim-shift', default='auto',
                      help='sim frame offset relative to replay, or "auto".')
  parser.add_argument('--max-auto-shift', type=int, default=3)
  parser.add_argument('--field-tolerance', type=float, default=1e-4)
  parser.add_argument('--context', type=int, default=8)
  parser.add_argument('--output-json', default='')
  parser.add_argument('--skip-policy', action='store_true')
  args = parser.parse_args()

  replay = _load_libmelee_replay(args.replay, args.frames)
  sim = _roll_sim_from_replay_inputs(replay)
  shift = (
      _best_shift(replay.games[0], sim.games[0], args.max_auto_shift)
      if args.sim_shift == 'auto'
      else int(args.sim_shift)
  )

  policy = None
  if not args.skip_policy:
    state = eval_lib.load_state(path=args.model_path)
    policy = tf_checkpoint.load_policy_from_tf_state(state, param_dtype='float32')
  perspective_results = []
  for perspective in (0, 1):
    replay_game, sim_game = _align_games(
        replay.games[perspective],
        sim.games[perspective],
        shift,
    )
    length = _game_length(replay_game)
    length = min(length, max(0, args.frames - args.start_frame))
    if args.start_frame:
      replay_game = _slice_game(replay_game, args.start_frame, args.start_frame + length)
      sim_game = _slice_game(sim_game, args.start_frame, args.start_frame + length)

    obs_stats = _compare_game_fields(
        replay_game,
        sim_game,
        tolerance=args.field_tolerance,
    )
    action_result = (
        _compare_policy_outputs(
            policy=policy,
            replay_game=replay_game,
            sim_game=sim_game,
        )
        if policy is not None
        else _empty_policy_result()
    )
    divergence = action_result['first_action_mismatch_frame']
    pre_end = length if divergence is None else max(0, int(divergence))
    pre_action_stats = _compare_game_fields(
        _slice_game(replay_game, 0, pre_end),
        _slice_game(sim_game, 0, pre_end),
        tolerance=args.field_tolerance,
    )
    perspective_results.append({
        'perspective': perspective,
        'self_port': replay.ports[perspective],
        'opponent_port': replay.ports[1 - perspective],
        'frames_compared': length,
        'first_action_mismatch_frame': divergence,
        'top_fields_all': obs_stats[:30],
        'top_fields_before_action_mismatch': pre_action_stats[:30],
        'policy_output': action_result,
        'context': _context_rows(
            replay_game=replay_game,
            sim_game=sim_game,
            action_result=action_result,
            center=divergence,
            radius=args.context,
        ),
    })

  result = {
      'replay': args.replay,
      'model_path': args.model_path,
      'ports': replay.ports,
      'stage': replay.stage.name,
      'characters': [character.name for character in replay.characters],
      'requested_frames': args.frames,
      'start_frame': args.start_frame,
      'sim_shift': shift,
      'perspectives': perspective_results,
  }
  text = json.dumps(_jsonify(result), indent=2, sort_keys=True)
  if args.output_json:
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  print(text)


class ReplayData(tuple):
  __slots__ = ()

  def __new__(cls, *, games, ports, stage, characters):
    return tuple.__new__(cls, (games, ports, stage, characters))

  @property
  def games(self):
    return self[0]

  @property
  def ports(self):
    return self[1]

  @property
  def stage(self):
    return self[2]

  @property
  def characters(self):
    return self[3]


def _load_libmelee_replay(path: str, frames: int) -> ReplayData:
  console = melee.Console(
      is_dolphin=False,
      allow_old_version=True,
      path=str(Path(path)),
  )
  console.connect()
  games = [[], []]
  ports = None
  stage = None
  characters = None
  try:
    while len(games[0]) < frames:
      gamestate = console.step()
      if gamestate is None:
        break
      if not gamestate.players:
        continue
      if ports is None:
        ports = sorted(gamestate.players)
        if len(ports) != 2:
          raise ValueError(f'expected singles replay with 2 ports, got {ports}')
        stage = gamestate.stage
        characters = tuple(gamestate.players[p].character for p in ports)
      if any(port not in gamestate.players for port in ports):
        continue
      games[0].append(parse_libmelee.get_game(gamestate, ports=(ports[0], ports[1])))
      games[1].append(parse_libmelee.get_game(gamestate, ports=(ports[1], ports[0])))
  finally:
    console.stop()
  if not games[0]:
    raise ValueError(f'no libmelee frames parsed from {path}')
  return ReplayData(
      games=[_stack_games(games[0]), _stack_games(games[1])],
      ports=ports,
      stage=stage,
      characters=characters,
  )


def _roll_sim_from_replay_inputs(replay: ReplayData) -> ReplayData:
  length = _game_length(replay.games[0])
  env = sim_env.SimBatchedEnvironment(
      num_envs=1,
      players={
          1: dolphin.AI(replay.characters[0]),
          2: dolphin.AI(replay.characters[1]),
      },
      length=max(128, length + 8),
      stage=replay.stage,
      character_pool=replay.characters,
  )
  games = [[], []]
  try:
    current = env.current_state(needs_reset=np.ones(1, dtype=np.bool_))
    games[0].append(_single_frame(current.gamestates[1], 0))
    games[1].append(_single_frame(current.gamestates[2], 0))
    for frame in range(max(0, length - 1)):
      output = env.step({
          1: _controller_frame(replay.games[0].p0.controller, frame + 1),
          2: _controller_frame(replay.games[0].p2.controller, frame + 1),
      })
      games[0].append(_single_frame(output.gamestates[1], 0))
      games[1].append(_single_frame(output.gamestates[2], 0))
  finally:
    env.stop()
  return ReplayData(
      games=[_stack_games(games[0]), _stack_games(games[1])],
      ports=replay.ports,
      stage=replay.stage,
      characters=replay.characters,
  )


def _stack_games(games):
  return utils.map_nt(lambda *xs: np.asarray(xs), *games)


def _single_frame(game, index: int):
  return utils.map_nt(lambda x: np.asarray(x)[index], game)


def _controller_frame(controller, index: int):
  return utils.map_nt(
      lambda x: np.asarray(x)[index:index + 1],
      controller,
  )


def _slice_game(game, start: int, end: int):
  return utils.map_nt(lambda x: np.asarray(x)[start:end], game)


def _game_length(game) -> int:
  return int(np.asarray(game.stage).shape[0])


def _align_games(replay_game, sim_game, shift: int):
  replay_len = _game_length(replay_game)
  sim_len = _game_length(sim_game)
  if shift >= 0:
    length = min(replay_len, sim_len - shift)
    return _slice_game(replay_game, 0, length), _slice_game(sim_game, shift, shift + length)
  offset = -shift
  length = min(replay_len - offset, sim_len)
  return _slice_game(replay_game, offset, offset + length), _slice_game(sim_game, 0, length)


def _best_shift(replay_game, sim_game, max_shift: int) -> int:
  if (
      _opening_alignment_score(*_align_games(replay_game, sim_game, 0))
      <= OPENING_ALIGNMENT_TOLERANCE
  ):
    return 0
  best = None
  for shift in range(-max_shift, max_shift + 1):
    try:
      a, b = _align_games(replay_game, sim_game, shift)
    except Exception:
      continue
    length = min(_game_length(a), 240)
    score = _alignment_score(_slice_game(a, 0, length), _slice_game(b, 0, length))
    if best is None or score < best[0]:
      best = (score, shift)
  if best is None:
    return 0
  return int(best[1])


def _opening_alignment_score(a, b) -> float:
  length = min(_game_length(a), _game_length(b), 24)
  if length <= 0:
    return float('inf')
  a = _slice_game(a, 0, length)
  b = _slice_game(b, 0, length)
  values = dict(_iter_leaves(a))
  other = dict(_iter_leaves(b))
  score = 0.0
  for field in ('stage', 'p0.action', 'p2.action', 'p0.y', 'p2.y'):
    x = np.asarray(values[field])
    y = np.asarray(other[field])
    if x.dtype.kind in 'f' or y.dtype.kind in 'f':
      score += float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
    else:
      score += float(np.mean(x != y)) * 100.0
  return score


def _alignment_score(a, b) -> float:
  opening_score = _opening_alignment_score(a, b)
  fields = (
      'p0.x', 'p0.y', 'p2.x', 'p2.y',
      'p0.percent', 'p2.percent',
      'p0.action', 'p2.action',
      'p0.stocks_left', 'p2.stocks_left',
  )
  values = dict(_iter_leaves(a))
  other = dict(_iter_leaves(b))
  score = 0.0
  for field in fields:
    x = np.asarray(values[field])
    y = np.asarray(other[field])
    if x.dtype.kind in 'f':
      score += float(np.nanmean(np.abs(x - y)))
    else:
      score += float(np.mean(x != y)) * 100.0
  return score + opening_score * 1000.0


def _iter_leaves(value, prefix=''):
  if isinstance(value, tuple) and hasattr(value, '_fields'):
    for field in value._fields:
      child = getattr(value, field)
      if isinstance(child, tuple) and not hasattr(child, '_fields') and len(child) == 0:
        continue
      yield from _iter_leaves(child, f'{prefix}.{field}' if prefix else field)
    return
  yield prefix, np.asarray(value)


def _compare_game_fields(a, b, *, tolerance: float):
  rows = []
  for path, x in _iter_leaves(a):
    y = dict(_iter_leaves(b))[path]
    if x.shape != y.shape:
      rows.append({
          'field': path,
          'shape_mismatch': [list(x.shape), list(y.shape)],
          'score': float('inf'),
      })
      continue
    if x.size == 0:
      continue
    if x.dtype.kind in 'f' or y.dtype.kind in 'f':
      diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
      bad = diff > tolerance
      first = int(np.argmax(bad)) if np.any(bad) else None
      rows.append({
          'field': path,
          'dtype': str(x.dtype),
          'kind': 'float',
          'max_abs': float(np.nanmax(diff)),
          'mean_abs': float(np.nanmean(diff)),
          'bad_frac': float(np.mean(bad)),
          'first_bad_frame': first,
          'replay_at_first': _json_scalar(x[first]) if first is not None else None,
          'sim_at_first': _json_scalar(y[first]) if first is not None else None,
          'score': float(np.nanmax(diff)) + 10.0 * float(np.mean(bad)),
      })
    else:
      bad = x != y
      first = int(np.argmax(bad)) if np.any(bad) else None
      rows.append({
          'field': path,
          'dtype': str(x.dtype),
          'kind': 'categorical',
          'mismatch_frac': float(np.mean(bad)),
          'first_bad_frame': first,
          'replay_at_first': _json_scalar(x[first]) if first is not None else None,
          'sim_at_first': _json_scalar(y[first]) if first is not None else None,
          'score': 100.0 * float(np.mean(bad)),
      })
  rows.sort(key=lambda r: (r.get('score', 0.0), r.get('max_abs', 0.0)), reverse=True)
  return rows


def _compare_policy_outputs(*, policy, replay_game, sim_game):
  length = min(_game_length(replay_game), _game_length(sim_game))
  replay_agent = jax_agents.BasicAgent(
      policy=policy,
      batch_size=1,
      name_code=0,
      seed=0,
      sample_kwargs=dict(temperature=1.0),
      compile=False,
      pack_args=False,
  )
  sim_agent = jax_agents.BasicAgent(
      policy=policy,
      batch_size=1,
      name_code=0,
      seed=0,
      sample_kwargs=dict(temperature=1.0),
      compile=False,
      pack_args=False,
  )

  per_frame = []
  first_mismatch = None
  for frame in range(length):
    needs_reset = np.array([frame == 0], dtype=np.bool_)
    replay_out = replay_agent.step_device(
        _slice_game(replay_game, frame, frame + 1),
        needs_reset,
    )
    sim_out = sim_agent.step_device(
        _slice_game(sim_game, frame, frame + 1),
        needs_reset,
    )
    replay_action = utils.map_nt(lambda x: np.asarray(x)[0], replay_out.controller_state)
    sim_action = utils.map_nt(lambda x: np.asarray(x)[0], sim_out.controller_state)
    action_diff = _action_diff(replay_action, sim_action)
    logits_max, logits_mean = _tree_abs_max_mean(replay_out.logits, sim_out.logits)
    row = {
        'frame': frame,
        'action_mismatch_fields': action_diff,
        'logits_max_abs': logits_max,
        'logits_mean_abs': logits_mean,
    }
    per_frame.append(row)
    if action_diff and first_mismatch is None:
      first_mismatch = frame
      break
  return {
      'first_action_mismatch_frame': first_mismatch,
      'frames_evaluated': len(per_frame),
      'first_mismatch_action_fields': (
          [] if first_mismatch is None else per_frame[-1]['action_mismatch_fields']),
      'last_logits_max_abs': per_frame[-1]['logits_max_abs'] if per_frame else 0.0,
      'last_logits_mean_abs': per_frame[-1]['logits_mean_abs'] if per_frame else 0.0,
      'per_frame_until_mismatch': per_frame,
  }


def _empty_policy_result():
  return {
      'first_action_mismatch_frame': None,
      'frames_evaluated': 0,
      'first_mismatch_action_fields': [],
      'last_logits_max_abs': 0.0,
      'last_logits_mean_abs': 0.0,
      'per_frame_until_mismatch': [],
  }


def _action_diff(a, b):
  fields = []
  for path, x in _iter_leaves(a):
    y = dict(_iter_leaves(b))[path]
    if np.asarray(x).item() != np.asarray(y).item():
      fields.append({
          'field': path,
          'replay': _json_scalar(x),
          'sim': _json_scalar(y),
      })
  return fields


def _tree_abs_max_mean(a, b):
  max_abs = 0.0
  total = 0.0
  count = 0
  for (_, x), (_, y) in zip(_iter_leaves(a), _iter_leaves(b)):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    if diff.size:
      max_abs = max(max_abs, float(np.nanmax(diff)))
      total += float(np.nansum(diff))
      count += int(diff.size)
  return max_abs, total / max(count, 1)


def _context_rows(*, replay_game, sim_game, action_result, center, radius: int):
  if center is None:
    center = min(_game_length(replay_game), action_result['frames_evaluated']) - 1
  start = max(0, int(center) - radius)
  end = min(_game_length(replay_game), int(center) + radius + 1)
  rows = []
  per_frame = {
      row['frame']: row for row in action_result['per_frame_until_mismatch']
  }
  for frame in range(start, end):
    row = {'frame': frame}
    for field in ('p0.x', 'p0.y', 'p2.x', 'p2.y', 'p0.action', 'p2.action',
                  'p0.percent', 'p2.percent', 'p0.on_ground', 'p2.on_ground'):
      replay_value = _get_leaf(replay_game, field)[frame]
      sim_value = _get_leaf(sim_game, field)[frame]
      row[field] = {
          'replay': _json_scalar(replay_value),
          'sim': _json_scalar(sim_value),
      }
    if frame in per_frame:
      row['policy'] = per_frame[frame]
    rows.append(row)
  return rows


def _get_leaf(value, path: str):
  for part in path.split('.'):
    value = getattr(value, part)
  return np.asarray(value)


def _json_scalar(value):
  value = np.asarray(value)
  if value.shape:
    value = value.item()
  else:
    value = value.item()
  if isinstance(value, np.generic):
    value = value.item()
  return value


def _jsonify(value):
  if isinstance(value, dict):
    return {str(k): _jsonify(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonify(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


if __name__ == '__main__':
  main()
