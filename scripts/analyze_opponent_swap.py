import argparse
from typing import NamedTuple

import numpy as np
import tensorflow as tf
import peppi_py

from slippi_ai import data as data_lib
from slippi_ai import saving
from slippi_ai import types
from slippi_ai import utils
from slippi_db import parse_peppi


class PortAssignment(NamedTuple):
  main: int
  teammate: int
  opp_a: int
  opp_b: int


def _team_key(player) -> tuple[int, int]:
  team = player.team
  return (int(team.color), int(team.shade))


def _infer_ports(game: peppi_py.Game, main_port: int) -> PortAssignment:
  players = list(game.start.players)
  if len(players) != 4:
    raise ValueError(f"Expected 4 players, got {len(players)}")

  if main_port < 0 or main_port >= 4:
    raise ValueError(f"main_port must be in [0,3], got {main_port}")

  main_team = _team_key(players[main_port])
  teammate = None
  opponents = []
  for idx, p in enumerate(players):
    if idx == main_port:
      continue
    if _team_key(p) == main_team:
      teammate = idx
    else:
      opponents.append(idx)

  if teammate is None:
    raise ValueError(f"Could not find teammate for main_port={main_port}")
  if len(opponents) != 2:
    raise ValueError(f"Expected 2 opponents, got {len(opponents)}")

  return PortAssignment(
      main=main_port,
      teammate=teammate,
      opp_a=opponents[0],
      opp_b=opponents[1],
  )


def _slice_nt(x, start: int, length: int):
  def slicer(arr):
    return arr[start:start + length]
  return utils.map_nt(slicer, x)


def _add_batch_dim(x):
  def add(arr):
    if not isinstance(arr, np.ndarray):
      raise TypeError(f"Expected np.ndarray, got {type(arr)}")
    if arr.ndim == 0:
      return arr.reshape((1,))
    return np.expand_dims(arr, axis=1)  # [T] -> [T, B=1]
  return utils.map_nt(add, x)


def _swap_p2_p3(game_nt: types.Game) -> types.Game:
  return game_nt._replace(p2=game_nt.p3, p3=game_nt.p2)


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
  a = a.reshape(-1).astype(np.float64, copy=False)
  b = b.reshape(-1).astype(np.float64, copy=False)
  denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
  return float(np.dot(a, b) / denom)


def _tensor_stats(x: tf.Tensor) -> dict[str, float]:
  x = tf.cast(x, tf.float32)
  return dict(
      mean=float(tf.reduce_mean(x).numpy()),
      abs_mean=float(tf.reduce_mean(tf.abs(x)).numpy()),
      rms=float(tf.sqrt(tf.reduce_mean(tf.square(x))).numpy()),
      max=float(tf.reduce_max(x).numpy()),
  )


def _diff_stats(a: tf.Tensor, b: tf.Tensor) -> dict[str, float]:
  diff = a - b
  a_stats = _tensor_stats(a)
  diff_stats = _tensor_stats(diff)
  rel = diff_stats["abs_mean"] / max(1e-12, a_stats["abs_mean"])
  return dict(
      abs_mean=diff_stats["abs_mean"],
      rms=diff_stats["rms"],
      max_abs=float(tf.reduce_max(tf.abs(diff)).numpy()),
      rel_abs_mean=rel,
  )

def _diff_stats_any(a, b) -> dict[str, float]:
  """Like _diff_stats but supports nested LSTM states."""
  flat_a = tf.nest.flatten(a)
  flat_b = tf.nest.flatten(b)
  if not flat_a:
    raise ValueError("Empty structure for diff stats")
  if len(flat_a) != len(flat_b):
    raise ValueError("Mismatched structures for diff stats")

  diffs = [
      tf.cast(x, tf.float32) - tf.cast(y, tf.float32)
      for x, y in zip(flat_a, flat_b)
  ]
  a_vec = tf.concat([tf.reshape(tf.cast(x, tf.float32), [-1]) for x in flat_a], axis=0)
  d_vec = tf.concat([tf.reshape(d, [-1]) for d in diffs], axis=0)

  abs_mean = float(tf.reduce_mean(tf.abs(d_vec)).numpy())
  rms = float(tf.sqrt(tf.reduce_mean(tf.square(d_vec))).numpy())
  max_abs = float(tf.reduce_max(tf.abs(d_vec)).numpy())
  rel_abs_mean = abs_mean / max(1e-12, float(tf.reduce_mean(tf.abs(a_vec)).numpy()))
  return dict(abs_mean=abs_mean, rms=rms, max_abs=max_abs, rel_abs_mean=rel_abs_mean)


def _get_p2_p3_slices(embed_game) -> tuple[slice, slice]:
  offset = 0
  p2_slice = None
  p3_slice = None
  for field, op in embed_game.embedding:
    size = int(op.size)
    if field == "p2":
      p2_slice = slice(offset, offset + size)
    if field == "p3":
      p3_slice = slice(offset, offset + size)
    offset += size
  if p2_slice is None or p3_slice is None:
    raise ValueError("Could not locate p2/p3 slices in embed_game")
  return p2_slice, p3_slice


def _windowed_rel_abs_mean(a: np.ndarray, b: np.ndarray, window: int, starts: np.ndarray) -> np.ndarray:
  """Compute rel_abs_mean per window for arrays shaped [T, ...]."""
  if a.shape != b.shape:
    raise ValueError("Shape mismatch for windowed stats")
  if a.shape[0] < window:
    raise ValueError("Window longer than sequence")

  # Flatten feature dims for cheap aggregate stats.
  a = a.reshape((a.shape[0], -1))
  b = b.reshape((b.shape[0], -1))
  dif = a - b

  out = np.zeros((len(starts),), dtype=np.float64)
  for i, s in enumerate(starts):
    e = int(s) + window
    denom = float(np.mean(np.abs(a[s:e]))) + 1e-12
    out[i] = float(np.mean(np.abs(dif[s:e]))) / denom
  return out


def _summarize_dist(x: np.ndarray) -> dict[str, float]:
  x = np.asarray(x, dtype=np.float64)
  return dict(
      mean=float(np.mean(x)),
      p50=float(np.percentile(x, 50)),
      p90=float(np.percentile(x, 90)),
      p99=float(np.percentile(x, 99)),
      max=float(np.max(x)),
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default="testmodels/latest_imitation_v12_5800k.pkl")
  parser.add_argument("--replay_path", action="append", default=None)
  parser.add_argument("--main_port", type=int, default=0)
  parser.add_argument("--start", type=int, default=0, help="single-window start (used when --num_windows=0)")
  parser.add_argument("--length", type=int, default=256, help="window length")
  parser.add_argument("--num_windows", type=int, default=0, help="if >0, sample this many random windows per replay")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  policy = saving.load_policy_from_disk(args.model_path)

  replay_paths = args.replay_path or ["runback_replay.slp"]

  print("**Model**")
  print(f"model_path={args.model_path}")
  print(f"embed_game.size={policy.embed_game.size} embed_state_action.size={policy.embed_state_action.size}")

  opponent_pooling = getattr(policy, "_opponent_pooling", None)
  if opponent_pooling is not None:
    print("\n**Opponent Pooling**")
    print(f"policy._opponent_pooling.config={getattr(opponent_pooling, '_config', None)}")

  print("\n**Encoder Weight Slice Similarity**")
  if opponent_pooling is not None and getattr(opponent_pooling, "_config", None) is not None and opponent_pooling._config.enabled:
    print("skipped: opponent pooling is enabled, so p2/p3 slice similarity in encoder weights is not meaningful.")
  else:
    encoder_w = None
    for v in policy.trainable_variables:
      if v.name == "encoder/w:0":
        encoder_w = v
        break
    if encoder_w is None:
      raise ValueError("Could not find policy encoder/w:0")

    p2_slice, p3_slice = _get_p2_p3_slices(policy.embed_game)
    w = encoder_w.numpy()
    w_p2 = w[p2_slice, :]
    w_p3 = w[p3_slice, :]
    flat_cos = _cosine_similarity(w_p2, w_p3)
    denom = (np.linalg.norm(w_p2, axis=0) * np.linalg.norm(w_p3, axis=0)) + 1e-12
    per_unit = (np.sum(w_p2 * w_p3, axis=0) / denom).astype(np.float64, copy=False)
    print(f"p2_slice={p2_slice} p3_slice={p3_slice} player_embed_size={w_p2.shape[0]} hidden_size={w_p2.shape[1]}")
    print(f"flat_cosine={flat_cos:.6g} per_unit(mean={float(np.mean(per_unit)):.6g}, p50={float(np.percentile(per_unit,50)):.6g}, p95={float(np.percentile(per_unit,95)):.6g})")

  rng = np.random.default_rng(args.seed)
  for replay_path in replay_paths:
    peppi_game = peppi_py.read_slippi(replay_path)
    if len(list(peppi_game.frames.ports)) != 4 or not peppi_game.start.is_teams:
      raise ValueError(f"{replay_path} must be a 4-player teams game for this analysis.")

    ports = _infer_ports(peppi_game, args.main_port)
    game_array = parse_peppi.from_peppi(peppi_game)
    full_game = types.game_array_to_nt(game_array)

    info = data_lib.ReplayInfo(
        path=replay_path,
        main_player_index=ports.main,
        teammate_index=ports.teammate,
        main_player_name="",
        meta=(),
        opponent_order=(ports.opp_a, ports.opp_b),
    )
    full_game = data_lib.swap_players(full_game, info)

    full_len = len(full_game.stage)
    if full_len < args.length:
      raise ValueError(f"{replay_path}: too short for length={args.length} (len={full_len})")

    controller = full_game.p0.controller
    name_codes = np.zeros([full_len], dtype=np.int32)
    sa = data_lib.StateAction(full_game, controller, name_codes)
    sa_s = data_lib.StateAction(_swap_p2_p3(full_game), controller, name_codes)
    sa = policy.embed_state_action.from_state(sa)
    sa_s = policy.embed_state_action.from_state(sa_s)
    sa = _add_batch_dim(sa)
    sa_s = _add_batch_dim(sa_s)

    inputs = policy.embed_state_action(sa)
    inputs_s = policy.embed_state_action(sa_s)
    if opponent_pooling is not None:
      inputs = opponent_pooling(inputs)
      inputs_s = opponent_pooling(inputs_s)
    reset = tf.zeros([full_len, 1], dtype=tf.bool)
    initial_state = policy.initial_state(1)

    outputs, _ = policy.network.scan(inputs, reset, initial_state)
    outputs_s, _ = policy.network.scan(inputs_s, reset, initial_state)
    values = tf.squeeze(policy.value_head(outputs), -1)
    values_s = tf.squeeze(policy.value_head(outputs_s), -1)

    out_np = tf.squeeze(outputs, 1).numpy()
    out_np_s = tf.squeeze(outputs_s, 1).numpy()
    val_np = tf.squeeze(values, 1).numpy()
    val_np_s = tf.squeeze(values_s, 1).numpy()

    if args.num_windows <= 0:
      if args.start < 0 or args.start + args.length > full_len:
        raise ValueError(f"{replay_path}: slice [{args.start},{args.start + args.length}) exceeds len {full_len}")
      starts = np.array([args.start], dtype=np.int32)
    else:
      max_start = full_len - args.length
      starts = rng.integers(low=0, high=max_start + 1, size=args.num_windows, dtype=np.int32)

    out_rel = _windowed_rel_abs_mean(out_np, out_np_s, args.length, starts)
    val_rel = _windowed_rel_abs_mean(val_np, val_np_s, args.length, starts)

    out_sum = _summarize_dist(out_rel)
    val_sum = _summarize_dist(val_rel)

    print("\n**Replay**")
    print(f"replay_path={replay_path}")
    print(f"ports(main,teammate,opp_a,opp_b)={ports}")
    print(f"length={full_len} window_length={args.length} windows={len(starts)} seed={args.seed}")
    print(f"trunk_outputs rel_abs_mean mean={out_sum['mean']:.6g} p50={out_sum['p50']:.6g} p90={out_sum['p90']:.6g} p99={out_sum['p99']:.6g} max={out_sum['max']:.6g}")
    print(f"value        rel_abs_mean mean={val_sum['mean']:.6g} p50={val_sum['p50']:.6g} p90={val_sum['p90']:.6g} p99={val_sum['p99']:.6g} max={val_sum['max']:.6g}")


if __name__ == "__main__":
  main()
