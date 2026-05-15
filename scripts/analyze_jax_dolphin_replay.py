import argparse
import json
from pathlib import Path

import jax
import melee
import numpy as np
import peppi_py

from slippi_ai import eval_lib
from slippi_ai import data
from slippi_ai import types
from slippi_ai import utils
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import tf_checkpoint
from slippi_db import parse_peppi


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--replay', required=True)
  parser.add_argument('--model', required=True)
  parser.add_argument('--frames', type=int, default=3000)
  parser.add_argument('--start-frame', type=int, default=123)
  parser.add_argument('--max-delay', type=int, default=40)
  parser.add_argument(
      '--likelihood-delays',
      default='',
      help='Comma-separated delays for teacher-forced likelihood; default uses 1..max-delay.',
  )
  parser.add_argument('--platform', choices=('cpu', 'gpu'), default='cpu')
  parser.add_argument(
      '--neutral-controller-observation',
      action='store_true',
      help='Force observed controller fields to neutral before policy inference.',
  )
  parser.add_argument('--output-json', default='')
  args = parser.parse_args()

  raw = peppi_py.read_slippi(args.replay)
  ports = list(raw.frames.ports)
  if len(ports) != 2:
    raise ValueError(f'expected singles replay with 2 ports, got {len(ports)}')

  players = [parse_peppi._player_from_port(port) for port in ports]
  length = min(len(players[0].x), int(args.frames))
  stage = melee.enums.to_internal_stage(raw.start.stage)

  state = eval_lib.load_state(path=args.model)
  policy = tf_checkpoint.load_policy_from_tf_state(state, param_dtype='float32')
  embed_controller = policy.controller_head.controller_embedding

  results = []
  for self_index, opponent_index in ((0, 1), (1, 0)):
    game = _perspective_game(
        self_player=players[self_index],
        opponent_player=players[opponent_index],
        stage=stage,
        length=length,
    )
    if args.neutral_controller_observation:
      game = _with_neutral_controller_observation(game, length)
    recorded = embed_controller.from_state(players[self_index].controller)
    recorded = utils.map_nt(lambda x: np.asarray(x)[:length], recorded)

    sampled = _sample_policy_labels(
        policy=policy,
        game=game,
        length=length,
        seed=0,
    )

    result = {
        'self_port_index': self_index,
        'opponent_port_index': opponent_index,
        'best_delays': _delay_scores(
            sampled=sampled,
            recorded=recorded,
            start_frame=args.start_frame,
            max_delay=args.max_delay,
        )[:10],
        'component_scores_at_policy_delay': _component_scores(
            sampled=sampled,
            recorded=recorded,
            start_frame=args.start_frame,
            delay=policy.delay,
        ),
        'teacher_forced_likelihoods': _teacher_forced_likelihood_scores(
            policy=policy,
            game=game,
            recorded=recorded,
            start_frame=args.start_frame,
            max_delay=args.max_delay,
            delays=_parse_delays(args.likelihood_delays),
        )[:10],
        'first_mismatches_at_policy_delay': _first_mismatches(
            sampled=sampled,
            recorded=recorded,
            start_frame=args.start_frame,
            delay=policy.delay,
            limit=8,
        ),
    }
    results.append(result)

  out = {
      'replay': args.replay,
      'model': args.model,
      'stage': stage.name,
      'length': length,
      'policy_delay': policy.delay,
      'results': results,
  }
  text = json.dumps(out, indent=2, sort_keys=True)
  if args.output_json:
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  print(text)


def _perspective_game(
    *,
    self_player: types.Player,
    opponent_player: types.Player,
    stage: melee.Stage,
    length: int,
) -> types.Game:
  empty = parse_peppi.zero_out_namedtuple(self_player)
  empty.is_dead[:length] = True
  empty = utils.map_nt(lambda x: x[:length], empty)

  frame_ids = np.arange(length, dtype=np.float32)
  return types.Game(
      p0=utils.map_nt(lambda x: x[:length], self_player),
      p1=empty,
      p2=utils.map_nt(lambda x: x[:length], opponent_player),
      p3=empty,
      stage=np.full(length, stage.value, dtype=np.uint8),
      randall_phase=np.mod(frame_ids, 1200).astype(np.float32),
      randall=parse_peppi._parse_randall(stage, np.arange(length, dtype=np.int32)),
      items=parse_peppi._empty_items(length),
      is_teams=np.zeros(length, dtype=np.bool_),
  )


def _neutral_controller_like(controller: types.Controller, length: int) -> types.Controller:
  shape = (int(length),)
  return types.Controller(
      main_stick=types.Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      c_stick=types.Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      shoulder=np.zeros(shape, dtype=np.float32),
      buttons=types.Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in types.Buttons._fields
      }),
  )


def _with_neutral_controller_observation(game: types.Game, length: int) -> types.Game:
  p0 = game.p0._replace(controller=_neutral_controller_like(game.p0.controller, length))
  p2 = game.p2._replace(controller=_neutral_controller_like(game.p2.controller, length))
  return game._replace(p0=p0, p2=p2)


def _sample_policy_labels(
    *,
    policy,
    game: types.Game,
    length: int,
    seed: int,
):
  agent = jax_agents.BasicAgent(
      policy=policy,
      batch_size=1,
      name_code=0,
      seed=seed,
      sample_kwargs=dict(temperature=1.0),
      compile=True,
      pack_args=True,
  )
  needs_reset = np.zeros(1, dtype=np.bool_)
  labels = []
  for frame in range(length):
    needs_reset[0] = frame == 0
    one_frame = utils.map_nt(lambda x, frame=frame: x[frame:frame + 1], game)
    action = agent.step_controller_state(one_frame, needs_reset)
    labels.append(utils.map_nt(lambda x: np.asarray(x)[0], action))
  jax.block_until_ready(agent._hidden_state)
  return _stack_labels(labels)


def _stack_labels(labels):
  return utils.map_nt(lambda *xs: np.asarray(xs), *labels)


def _flatten_labels(labels):
  leaves = []
  names = []

  def visit(prefix, value):
    if isinstance(value, tuple) and hasattr(value, '_fields'):
      for field in value._fields:
        visit(f'{prefix}.{field}' if prefix else field, getattr(value, field))
      return
    names.append(prefix)
    leaves.append(np.asarray(value))

  visit('', labels)
  return names, leaves


def _delay_scores(*, sampled, recorded, start_frame: int, max_delay: int):
  _, sampled_leaves = _flatten_labels(sampled)
  _, recorded_leaves = _flatten_labels(recorded)
  scores = []
  for delay in range(max_delay + 1):
    end = min(*(len(x) for x in sampled_leaves + recorded_leaves)) - delay
    if end <= start_frame:
      continue
    total = 0
    equal = 0
    exact_frames = None
    for s, r in zip(sampled_leaves, recorded_leaves):
      a = s[start_frame:end]
      b = r[start_frame + delay:end + delay]
      total += a.size
      equal += int(np.count_nonzero(a == b))
      same = a == b
      exact_frames = same if exact_frames is None else np.logical_and(exact_frames, same)
    scores.append({
        'delay': delay,
        'leaf_equal_frac': equal / max(total, 1),
        'exact_controller_frac': float(np.mean(exact_frames)) if exact_frames is not None else 0.0,
    })
  scores.sort(key=lambda x: (x['exact_controller_frac'], x['leaf_equal_frac']), reverse=True)
  return scores


def _component_scores(*, sampled, recorded, start_frame: int, delay: int):
  names, sampled_leaves = _flatten_labels(sampled)
  _, recorded_leaves = _flatten_labels(recorded)
  end = min(*(len(x) for x in sampled_leaves + recorded_leaves)) - delay
  out = {}
  for name, s, r in zip(names, sampled_leaves, recorded_leaves):
    if end <= start_frame:
      out[name] = None
      continue
    out[name] = float(np.mean(s[start_frame:end] == r[start_frame + delay:end + delay]))
  return out


def _first_mismatches(*, sampled, recorded, start_frame: int, delay: int, limit: int):
  names, sampled_leaves = _flatten_labels(sampled)
  _, recorded_leaves = _flatten_labels(recorded)
  end = min(*(len(x) for x in sampled_leaves + recorded_leaves)) - delay
  mismatches = []
  for frame in range(start_frame, end):
    bad = []
    for name, s, r in zip(names, sampled_leaves, recorded_leaves):
      sv = s[frame]
      rv = r[frame + delay]
      if sv != rv:
        bad.append({'component': name, 'sampled': int(sv), 'recorded': int(rv)})
    if bad:
      mismatches.append({
          'sample_frame': frame,
          'recorded_frame': frame + delay,
          'components': bad,
      })
      if len(mismatches) >= limit:
        break
  return mismatches


def _teacher_forced_likelihood_scores(
    *,
    policy,
    game: types.Game,
    recorded,
    start_frame: int,
    max_delay: int,
    delays: list[int] | None,
):
  scores = []
  if delays is None:
    delays = list(range(1, max_delay + 1))
  for delay in delays:
    length = min(
        len(game.stage) - delay,
        min(len(x) for x in _flatten_labels(recorded)[1]) - delay,
    )
    if length <= start_frame:
      continue

    state = utils.map_nt(lambda x, length=length: x[:length + 1], game)
    state = utils.map_nt(lambda x: np.expand_dims(x, axis=1), state)
    action = utils.map_nt(
        lambda x, delay=delay, length=length: x[delay - 1:delay + length],
        recorded,
    )
    action = utils.map_nt(lambda x: np.expand_dims(x, axis=1), action)
    encoded_state = policy.network.encode_game(state)
    frames = data.Frames(
        state_action=data.StateAction(
            state=encoded_state,
            action=action,
            name=np.zeros((length + 1, 1), dtype=data.NAME_DTYPE),
        ),
        is_resetting=_reset_array(length + 1),
        reward=np.zeros((length + 1, 1), dtype=np.float32),
    )
    out = policy.unroll_with_outputs(frames, policy.initial_state((1,)))
    distances = jax.device_get(out.distances.distance)
    logits = jax.device_get(out.distances.logits)
    target = utils.map_nt(lambda x: x[1:], action)

    component_nll = {}
    component_argmax = {}
    total_nll = None
    for name, dist, logit, target_leaf in _iter_components(distances, logits, target):
      dist = np.asarray(dist).squeeze(axis=1)
      logit = np.asarray(logit).squeeze(axis=1)
      target_leaf = np.asarray(target_leaf).squeeze(axis=1)
      window = slice(start_frame, None)
      component_nll[name] = float(np.mean(dist[window]))
      component_argmax[name] = float(np.mean(
          np.argmax(logit[window], axis=-1) == target_leaf[window]))
      total_nll = dist if total_nll is None else total_nll + dist
    scores.append({
        'delay': delay,
        'mean_nll': float(np.mean(total_nll[start_frame:])),
        'component_nll': component_nll,
        'component_argmax_frac': component_argmax,
    })

  scores.sort(key=lambda x: x['mean_nll'])
  return scores


def _parse_delays(value: str) -> list[int] | None:
  if not value.strip():
    return None
  return [int(part) for part in value.split(',') if part.strip()]


def _reset_array(length: int):
  reset = np.zeros((length, 1), dtype=np.bool_)
  reset[0, 0] = True
  return reset


def _iter_components(distances, logits, target):
  out = []

  def visit(prefix, d, l, t):
    if isinstance(d, tuple) and hasattr(d, '_fields'):
      for field in d._fields:
        visit(
            f'{prefix}.{field}' if prefix else field,
            getattr(d, field),
            getattr(l, field),
            getattr(t, field),
        )
      return
    out.append((prefix, d, l, t))

  visit('', distances, logits, target)
  return out


if __name__ == '__main__':
  main()
