"""Visualize value-function predictions over a Slippi replay.

This script loads a saved model checkpoint, replays a provided `.slp` file
from the perspective of a chosen player (and optional teammate), and records
the predicted discounted returns alongside the actual rewards accrued during
the match.

Outputs include per-frame values and a PNG plot comparing the following
signals over time:
  * Model-predicted value (discounted future reward)
  * Realized discounted return (using the same discount factor)
  * Realized cumulative reward (undiscounted)

Example usage:

```
python scripts/value_trace.py \
  --model_path=experiments/doubles_delay_18/latest.pkl \
  --replay_path=path/to/match.slp \
  --main_port=1
```
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

from absl import app
from absl import flags
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tree

import peppi_py

from slippi_ai import data as data_lib
from slippi_ai import reward
from slippi_ai import rl_lib
from slippi_ai import saving
from slippi_ai import types
from slippi_ai import value_function as vf_lib

from slippi_db import parse_peppi


FLAGS = flags.FLAGS


def _normalize_port(raw_port) -> int:
  """Normalizes port representations to 1-based integers (1-4).

  Slippi metadata can encode ports as strings like 'P1', plain numerals, or
  zero-based integers. This helper converts the common variants into the
  1-based integer form expected by the flags and downstream logic.
  """

  if raw_port is None:
    raise ValueError('Encountered missing port in replay metadata.')

  if isinstance(raw_port, str):
    stripped = raw_port.strip().upper()
    if stripped.startswith('P'):
      stripped = stripped[1:]
    if not stripped:
      raise ValueError(f'Could not parse port from string {raw_port!r}.')
    try:
      raw_port = int(stripped, 10)
    except ValueError as exc:
      raise ValueError(f'Could not parse port from string {raw_port!r}.') from exc

  if isinstance(raw_port, (np.integer, int)):
    port_int = int(raw_port)
    if 1 <= port_int <= 4:
      return port_int
    if 0 <= port_int <= 3:
      return port_int + 1

  raise ValueError(f'Unsupported port value: {raw_port!r}.')


def _infer_teammate_port(start_meta: dict, main_port: int) -> Optional[int]:
  """Attempts to infer the teammate port for doubles games."""
  is_teams = start_meta.get('is_teams') or start_meta.get('isTeams') or False
  if not is_teams:
    return None

  normalized_main_port = _normalize_port(main_port)
  main_team = None
  for player in start_meta.get('players', []):
    try:
      player_port = _normalize_port(player.get('port'))
    except ValueError:
      continue
    if player_port == normalized_main_port:
      main_team = player.get('team')
      break

  if main_team is None:
    return None

  for player in start_meta.get('players', []):
    try:
      port = _normalize_port(player.get('port'))
    except ValueError:
      continue
    if port == normalized_main_port:
      continue
    if player.get('team') == main_team:
      return port

  return None


def _make_empty_player(template: types.Player) -> types.Player:
  """Creates an empty (dead) player matching the template's structure."""
  empty = parse_peppi.zero_out_namedtuple(template)
  true_mask = np.ones_like(template.is_dead, dtype=np.bool_)
  zero_uint8 = np.zeros_like(template.stocks_left, dtype=np.uint8)
  return empty._replace(is_dead=true_mask, stocks_left=zero_uint8)


def _reorder_game(
    game: types.Game,
    port_order: Sequence[int],
    main_port: int,
    teammate_port: Optional[int],
) -> types.Game:
  """Returns a new Game with p0/p1 as the focus team and p2/p3 as opponents."""
  normalized_order = [_normalize_port(port) for port in port_order]
  normalized_main = _normalize_port(main_port)
  player_list = [game.p0, game.p1, game.p2, game.p3]
  port_to_index = {port: idx for idx, port in enumerate(normalized_order)}

  if normalized_main not in port_to_index:
    raise ValueError(
        f'Port {normalized_main} not present in replay (found {normalized_order}).')

  main_index = port_to_index[normalized_main]
  teammate_index = None
  if teammate_port is not None:
    normalized_teammate = _normalize_port(teammate_port)
    if normalized_teammate not in port_to_index:
      raise ValueError(
          f'Teammate port {normalized_teammate} not present in replay '
          f'(found {normalized_order}).')
    teammate_index = port_to_index[normalized_teammate]

  empty_player = _make_empty_player(player_list[main_index])

  ordered: list[types.Player] = []
  ordered.append(player_list[main_index])

  if teammate_index is not None:
    ordered.append(player_list[teammate_index])
  else:
    ordered.append(empty_player)

  for idx in range(4):
    if idx == main_index or idx == teammate_index:
      continue
    ordered.append(player_list[idx])

  while len(ordered) < 4:
    ordered.append(empty_player)

  return game._replace(p0=ordered[0], p1=ordered[1], p2=ordered[2], p3=ordered[3])


def _build_frames(
    game: types.Game,
    damage_ratio: float,
    name_code: int = 0,
) -> data_lib.Frames:
  """Converts a reordered Game into data.Frames with numpy arrays."""
  rewards = reward.compute_rewards(game, damage_ratio=damage_ratio)
  length = game.stage.shape[0]

  name_codes = np.full([length], name_code, np.int32)
  is_resetting = np.zeros([length], np.bool_)
  is_resetting[0] = True

  state_action = data_lib.StateAction(
      state=game,
      action=game.p0.controller,
      name=name_codes,
  )

  return data_lib.Frames(
      state_action=state_action,
      is_resetting=is_resetting,
      reward=rewards,
  )


def _embed_frames(
    frames: data_lib.Frames,
    embed_state_action,
) -> data_lib.Frames:
  """Embeds state/action and adds batch dimension expected by learners."""

  embedded = embed_state_action.from_state(frames.state_action)

  def expand(t):
    return tf.expand_dims(tf.convert_to_tensor(t), axis=1)

  embedded_state = tf.nest.map_structure(expand, embedded.state)
  embedded_action = tf.nest.map_structure(expand, embedded.action)
  embedded_name = expand(embedded.name)

  is_resetting = expand(frames.is_resetting.astype(np.bool_))
  rewards = expand(frames.reward.astype(np.float32))

  return data_lib.Frames(
      state_action=data_lib.StateAction(
          state=embedded_state,
          action=embedded_action,
          name=embedded_name,
      ),
      is_resetting=is_resetting,
      reward=rewards,
  )


def _load_value_function(
    state: dict,
    config: dict,
    policy,
) -> Optional[vf_lib.ValueFunction]:
  vf_config = config.get('value_function', {})
  if not vf_config.get('train_separate_network', False):
    return None

  if vf_config.get('separate_network_config', False):
    network_config = vf_config.get('network', {})
  else:
    network_config = config.get('network', {})

  value_fn = vf_lib.ValueFunction(
      network_config=network_config,
      embed_state_action=policy.embed_state_action,
  )

  # Initialize variables before assignment.
  dummy_state_action = policy.embed_state_action.dummy([2, 1])
  dummy_frames = data_lib.Frames(
      state_action=dummy_state_action,
      is_resetting=tf.zeros([2, 1], tf.bool),
      reward=tf.zeros([1, 1], tf.float32),
  )
  _ = value_fn.loss(
      frames=dummy_frames,
      initial_state=value_fn.initial_state(1),
      discount=0.99,
  )

  variables = state['state'].get('value_function')
  if not variables:
    raise ValueError('Checkpoint does not contain value function parameters.')

  tree.map_structure(lambda var, val: var.assign(val), value_fn.variables, variables)
  return value_fn


def _compute_discount(config: dict) -> float:
  learner_cfg = config.get('learner', {})
  halflife = learner_cfg.get('reward_halflife', 4)
  return float(0.5 ** (1 / (halflife * 60)))


def _get_damage_ratio(config: dict) -> float:
  data_cfg = config.get('data', {})
  return float(data_cfg.get('damage_ratio', 0.01))


def _prepare_outputs(
    rewards: np.ndarray,
    discount: float,
    predicted_values: np.ndarray,
) -> dict[str, np.ndarray]:
  rewards_tf = tf.convert_to_tensor(rewards, tf.float32)
  discounts = tf.ones_like(rewards_tf) * tf.convert_to_tensor(discount, tf.float32)
  discounted_returns = rl_lib.discounted_returns(
      rewards=rewards_tf,
      discounts=discounts,
      bootstrap=tf.constant(0.0, tf.float32),
  ).numpy()

  cumulative_reward = np.cumsum(rewards, dtype=np.float32)
  return dict(
      predicted=predicted_values,
      discounted=discounted_returns,
      cumulative=cumulative_reward,
  )


def _write_csv(csv_path: Path, time_axis: np.ndarray, traces: dict[str, np.ndarray]):
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  with csv_path.open('w', newline='') as f:
    writer = csv.writer(f)
    header = ['frame', 'time_sec'] + list(traces.keys())
    writer.writerow(header)
    for idx, t in enumerate(time_axis):
      row = [idx, t]
      for key in traces:
        row.append(traces[key][idx])
      writer.writerow(row)


def _plot_traces(
    plot_path: Path,
    time_axis: np.ndarray,
    traces: dict[str, np.ndarray],
    title: str,
):
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  plt.figure(figsize=(10, 5))
  plt.plot(time_axis, traces['predicted'], label='Predicted value')
  plt.plot(time_axis, traces['discounted'], label='Actual discounted return', linestyle='--')
  plt.plot(time_axis, traces['cumulative'], label='Cumulative reward', linestyle=':')
  plt.xlabel('Time (s)')
  plt.ylabel('Team value estimate')
  plt.title(title)
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_path)
  plt.close()


def main(_):
  model_path = Path(FLAGS.model_path)
  replay_path = Path(FLAGS.replay_path)

  if not model_path.exists():
    raise FileNotFoundError(f'Model path not found: {model_path}')
  if not replay_path.exists():
    raise FileNotFoundError(f'Replay path not found: {replay_path}')

  state = saving.load_state_from_disk(str(model_path))
  config = state['config']

  policy = saving.load_policy_from_state(state)

  value_function = _load_value_function(state, config, policy)
  discount = _compute_discount(config)
  damage_ratio = _get_damage_ratio(config)

  raw_game = peppi_py.read_slippi(str(replay_path))
  game_struct = parse_peppi.from_peppi(raw_game)
  game = types.game_array_to_nt(game_struct)

  ports = sorted(player.get('port') for player in raw_game.start['players'])
  teammate_port = FLAGS.teammate_port
  if teammate_port is None:
    teammate_port = _infer_teammate_port(raw_game.start, FLAGS.main_port)

  game_team = _reorder_game(game, ports, FLAGS.main_port, teammate_port)

  frames_np = _build_frames(game_team, damage_ratio)
  frames = _embed_frames(frames_np, policy.embed_state_action)

  batch_size = 1

  if value_function is not None:
    outputs, _ = value_function.loss(
        frames=frames,
        initial_state=value_function.initial_state(batch_size),
        discount=discount,
    )
    value_outputs = outputs
  else:
    unroll_outputs = policy.unroll(
        frames=frames,
        initial_state=policy.initial_state(batch_size),
        discount=discount,
    )
    value_outputs = unroll_outputs.value_outputs

  returns = value_outputs.returns.numpy().squeeze(axis=-1)
  advantages = value_outputs.advantages.numpy().squeeze(axis=-1)
  predicted_values = returns - advantages

  rewards = frames_np.reward.astype(np.float32)
  traces = _prepare_outputs(rewards, discount, predicted_values)

  frames_axis = np.arange(len(predicted_values), dtype=np.float32)
  time_axis = frames_axis / 60.0

  plot_path = Path(FLAGS.plot_path) if FLAGS.plot_path else replay_path.with_suffix('.value.png')
  title = f'{replay_path.name} (port {FLAGS.main_port})'
  _plot_traces(plot_path, time_axis, traces, title)

  if FLAGS.csv_path:
    csv_path = Path(FLAGS.csv_path)
  else:
    csv_path = replay_path.with_suffix('.value.csv')
  _write_csv(csv_path, time_axis, traces)

  print('Saved plot to', plot_path)
  print('Saved per-frame data to', csv_path)
  print('Final predicted value:', float(traces['predicted'][-1]))
  print('Final discounted return:', float(traces['discounted'][-1]))
  print('Final cumulative reward:', float(traces['cumulative'][-1]))


def _define_flags():
  flags.DEFINE_string('model_path', None, 'Path to pickled model checkpoint (*.pkl).')
  flags.DEFINE_string('replay_path', None, 'Path to Slippi replay (*.slp).')
  flags.DEFINE_integer('main_port', 1, 'Port (1-4) for the player perspective.')
  flags.DEFINE_integer(
      'teammate_port',
      None,
      'Optional teammate port. If omitted for teams, inferred from metadata.',
  )
  flags.DEFINE_string(
      'plot_path',
      None,
      'Optional output path for the PNG plot (defaults next to replay).',
  )
  flags.DEFINE_string(
      'csv_path',
      None,
      'Optional output path for per-frame CSV data (defaults next to replay).',
  )

  flags.mark_flag_as_required('model_path')
  flags.mark_flag_as_required('replay_path')


if __name__ == '__main__':
  _define_flags()
  app.run(main)
