"""Compact reward computation for sim rollouts.

The generic reward path accepts a full libmelee-shaped `Game` transition tree.
For sim rollouts that is wasteful: reward only reads a small fixed set of player
fields, and terminal correction only needs to patch the next-frame side of
those fields on reset lanes. This module keeps the same reward semantics while
avoiding construction of a full `[current, next]` Game tree.
"""

import dataclasses

import melee
import numpy as np

from slippi_ai import reward as reward_lib
from slippi_ai import utils


_PLAYER_NAMES = ('p0', 'p1', 'p2', 'p3')
_TEAM1 = ('p0', 'p1')
_TEAM2 = ('p2', 'p3')
_DYING_ACTION_MAX = 0xA
_EDGE_CATCHING = melee.Action.EDGE_CATCHING.value
_ZELDA = melee.Character.ZELDA.value


@dataclasses.dataclass(frozen=True)
class TerminalRewardOverride:
  transition_index: int
  reset_mask: np.ndarray
  terminal_game: object


def masked_numpy_tree(value, mask: np.ndarray):
  mask = np.asarray(mask, dtype=np.bool_)
  return utils.map_single_structure(lambda x: np.asarray(x)[mask].copy(), value)


def compute_transition_rewards(
    time_major_game,
    *,
    terminal_reward_overrides: list[TerminalRewardOverride],
    reward_config: reward_lib.RewardConfig,
) -> np.ndarray:
  """Compute `[T, B]` rewards without building full transition Games."""

  fields = _collect_fields(
      time_major_game,
      terminal_reward_overrides,
      reward_config,
  )
  team1_reward = _team_reward(fields, _TEAM1, _TEAM2, reward_config)
  team2_reward = _team_reward(fields, _TEAM2, _TEAM1, reward_config)
  rewards = team1_reward - team2_reward
  rewards = _apply_mode_scale(
      rewards,
      _next_field(fields['is_teams']),
      reward_config,
  ).astype(np.float32)
  if not np.all(np.isfinite(rewards)):
    raise ValueError('Reward calculation produced non-finite values.')
  return rewards


def _collect_fields(game, terminal_reward_overrides, config: reward_lib.RewardConfig):
  player_fields = {'percent', 'action', 'stocks_left'}
  if config.ledge_grab_penalty:
    player_fields.update(('x', 'invulnerable'))
  if config.stalling_penalty:
    player_fields.update(('x', 'y'))
  if config.zelda_penalty:
    player_fields.update(('character', 'on_ground'))
  if config.approaching_factor:
    player_fields.update(('x', 'y'))

  fields = {}
  for player_name in _PLAYER_NAMES:
    player = getattr(game, player_name)
    fields[player_name] = {
        field_name: _field_with_next_override(
            getattr(player, field_name),
            terminal_reward_overrides,
            player_name,
            field_name,
        )
        for field_name in player_fields
    }
  if config.stalling_penalty:
    fields['stage'] = _field_with_next_override(
        game.stage, terminal_reward_overrides, None, 'stage')
  fields['is_teams'] = _field_with_next_override(
      game.is_teams, terminal_reward_overrides, None, 'is_teams')
  return fields


def _field_with_next_override(
    values,
    terminal_reward_overrides: list[TerminalRewardOverride],
    player_name: str | None,
    field_name: str,
):
  current = np.asarray(values)[:-1]
  next_values = np.asarray(values)[1:]
  if terminal_reward_overrides:
    next_values = np.array(next_values, copy=True)
    for override in terminal_reward_overrides:
      terminal_source = override.terminal_game
      if player_name is not None:
        terminal_source = getattr(terminal_source, player_name)
      terminal_values = np.asarray(getattr(terminal_source, field_name))
      next_values[override.transition_index, override.reset_mask] = terminal_values
  return current, next_values


def _current_field(field):
  return field[0]


def _next_field(field):
  return field[1]


def _team_reward(
    fields: dict,
    team: tuple[str, ...],
    opponents: tuple[str, ...],
    config: reward_lib.RewardConfig,
) -> np.ndarray:
  deaths = None
  damages = None
  ledges = None
  stalling = None
  zelda = None
  approach = None

  for player_name in team:
    player = fields[player_name]
    deaths = _add(deaths, _deaths(
        player,
        weight_by_percent=config.weight_deaths_by_percent,
    ))
    damages = _add(damages, _damages(player))
    if config.ledge_grab_penalty:
      ledges = _add(ledges, _bad_ledge_grabs(fields, player_name, opponents))
    if config.stalling_penalty:
      stalling = _add(stalling, _stalling(fields, player, config.stalling_threshold))
    if config.zelda_penalty:
      zelda = _add(zelda, _zelda_frames(player))
    if config.approaching_factor:
      approach = _add(approach, _approach(fields, player_name, opponents))

  normalization = (
      float(_count_active_players(fields, team))
      if config.team_size_normalization else float(len(team)))
  normalization = normalization or 1.0

  penalty = np.zeros_like(deaths, dtype=np.float32)
  penalty += deaths
  penalty += config.damage_ratio * damages
  if config.ledge_grab_penalty and ledges is not None:
    penalty += config.ledge_grab_penalty * ledges
  if config.stalling_penalty and stalling is not None:
    penalty += (config.stalling_penalty / 60.0) * stalling
  if config.zelda_penalty and zelda is not None:
    penalty += config.zelda_penalty * zelda
  penalty /= normalization

  team_reward = -penalty
  team_reward -= _stock_penalties(fields, team)
  if config.approaching_factor and approach is not None:
    team_reward += (config.approaching_factor * approach) / normalization
  return team_reward.astype(np.float32)


def _add(total: np.ndarray | None, value: np.ndarray) -> np.ndarray:
  value = np.asarray(value, dtype=np.float32)
  if total is None:
    return value
  return total + value


def _deaths(player: dict, *, weight_by_percent: bool) -> np.ndarray:
  current_dying = _current_field(player['action']) <= _DYING_ACTION_MAX
  next_dying = _next_field(player['action']) <= _DYING_ACTION_MAX
  death_frames = np.logical_and(np.logical_not(current_dying), next_dying).astype(
      np.float32)
  if weight_by_percent:
    percent = _next_field(player['percent']).astype(np.float32)
    death_frames *= 100.0 / (100.0 + percent)
  return death_frames


def _damages(player: dict) -> np.ndarray:
  current = _current_field(player['percent']).astype(np.float32)
  next_values = _next_field(player['percent']).astype(np.float32)
  return np.maximum(next_values - current, 0.0)


def _stock_penalties(fields: dict, team: tuple[str, ...]) -> np.ndarray:
  current = sum(
      _current_field(fields[player_name]['stocks_left']).astype(np.float32)
      for player_name in team)
  next_values = sum(
      _next_field(fields[player_name]['stocks_left']).astype(np.float32)
      for player_name in team)
  second_last = np.where(
      (current == 2) & (next_values == 1), 1.5, 0.0).astype(np.float32)
  last = np.where(
      (current == 1) & (next_values == 0), 3.5, 0.0).astype(np.float32)
  return second_last + last


def _bad_ledge_grabs(
    fields: dict,
    player_name: str,
    opponents: tuple[str, ...],
) -> np.ndarray:
  player = fields[player_name]
  ledge_grabs = np.logical_and(
      _current_field(player['action']) != _EDGE_CATCHING,
      _next_field(player['action']) == _EDGE_CATCHING,
  )
  if not opponents:
    return np.zeros_like(ledge_grabs, dtype=np.float32)

  player_x = _full_field(player['x'])
  center_direction = player_x < 0
  opponent_directions = [
      player_x < _full_field(fields[opponent]['x']) for opponent in opponents]
  opponents_towards_center = np.logical_and.reduce(
      [direction == center_direction for direction in opponent_directions])
  bad_ledge_grabs = np.logical_and(ledge_grabs, opponents_towards_center[:-1])

  invulnerable = np.zeros_like(_full_field(player['invulnerable']), dtype=bool)
  for opponent in opponents:
    invulnerable = np.logical_or(
        invulnerable,
        _full_field(fields[opponent]['invulnerable']),
    )
  bad_ledge_grabs = np.logical_and(
      bad_ledge_grabs,
      np.logical_not(invulnerable[:-1]),
  )
  return bad_ledge_grabs.astype(np.float32)


def _stalling(fields: dict, player: dict, threshold: float) -> np.ndarray:
  stage_xs = reward_lib.get_edge_x(_current_field(fields['stage'])[0])
  x = _next_field(player['x']).astype(np.float32)
  y = _next_field(player['y']).astype(np.float32)
  dx = np.maximum(np.abs(x) - stage_xs, 0)
  below = -np.minimum(y, 0)
  above = np.maximum(y - reward_lib.MAX_STALLING_Y, 0)
  dy = np.maximum(below, above)
  return (np.sqrt(np.square(dx) + np.square(dy)) > threshold).astype(np.float32)


def _zelda_frames(player: dict) -> np.ndarray:
  return np.logical_and(
      _next_field(player['character']) == _ZELDA,
      _next_field(player['on_ground']).astype(bool),
  ).astype(np.float32)


def _approach(
    fields: dict,
    player_name: str,
    opponents: tuple[str, ...],
) -> np.ndarray:
  if not opponents:
    return np.zeros_like(_current_field(fields[player_name]['x']), dtype=np.float32)

  player = fields[player_name]
  player_x = _current_field(player['x']).astype(np.float32)
  player_y = _current_field(player['y']).astype(np.float32)
  next_player_x = _next_field(player['x']).astype(np.float32)
  next_player_y = _next_field(player['y']).astype(np.float32)
  player_xy = np.stack([player_x, player_y], axis=-1)
  next_player_xy = np.stack([next_player_x, next_player_y], axis=-1)
  velocity = next_player_xy - player_xy

  opponent_x = np.stack([
      _current_field(fields[opponent]['x']).astype(np.float32)
      for opponent in opponents
  ], axis=0)
  opponent_y = np.stack([
      _current_field(fields[opponent]['y']).astype(np.float32)
      for opponent in opponents
  ], axis=0)
  opponent_active = np.stack([
      _current_field(fields[opponent]['stocks_left']) > 0
      for opponent in opponents
  ], axis=0)

  active_count = np.sum(opponent_active, axis=0).astype(np.float32)
  safe_count = np.maximum(active_count, 1.0)
  centroid_x = (
      np.sum(np.where(opponent_active, opponent_x, 0.0), axis=0) / safe_count)
  centroid_y = (
      np.sum(np.where(opponent_active, opponent_y, 0.0), axis=0) / safe_count)
  opponent_xy = np.stack([centroid_x, centroid_y], axis=-1)
  direction = _normalize(opponent_xy - player_xy)
  approach = np.sum(velocity * direction, axis=-1)

  current_dying = _current_field(player['action']) <= _DYING_ACTION_MAX
  next_dying = _next_field(player['action']) <= _DYING_ACTION_MAX
  respawning = np.logical_and(current_dying, np.logical_not(next_dying))
  approach = np.where(respawning, 0.0, approach)

  has_active_opponent = active_count > 0
  approach = np.where(has_active_opponent, approach, 0.0)

  if len(opponents) >= 2:
    inf = np.full_like(opponent_x, np.inf, dtype=np.float32)
    neg_inf = np.full_like(opponent_x, -np.inf, dtype=np.float32)
    min_x = np.min(np.where(opponent_active, opponent_x, inf), axis=0)
    max_x = np.max(np.where(opponent_active, opponent_x, neg_inf), axis=0)
    at_least_two_active = active_count >= 2
    between = np.logical_and(player_x >= min_x, player_x <= max_x)
    gated = np.logical_and(at_least_two_active, between)
    approach = np.where(gated, 0.0, approach)

  return approach.astype(np.float32)


def _normalize(xys, epsilon=1e-6):
  radius = np.sqrt(np.sum(np.square(xys), axis=-1, keepdims=True))
  return xys / (radius + epsilon)


def _full_field(field):
  return np.concatenate([_current_field(field), _next_field(field)[-1:]], axis=0)


def _count_active_players(fields: dict, team: tuple[str, ...]) -> int:
  count = 0
  for player_name in team:
    player = fields[player_name]
    if (
        np.any(_full_field(player['stocks_left']) > 0)
        or np.any(_full_field(player['percent']) > 0)
    ):
      count += 1
  return max(count, 1)


def _apply_mode_scale(
    rewards: np.ndarray,
    is_teams_next: np.ndarray,
    config: reward_lib.RewardConfig,
) -> np.ndarray:
  if config.singles_scale is None and config.doubles_scale is None:
    return rewards

  mode = np.asarray(is_teams_next, dtype=bool)
  while mode.ndim < rewards.ndim:
    mode = np.expand_dims(mode, axis=-1)
  mode = np.broadcast_to(mode, rewards.shape)

  if config.singles_scale is not None:
    rewards = np.where(mode, rewards, rewards * config.singles_scale)
  if config.doubles_scale is not None:
    rewards = np.where(mode, rewards * config.doubles_scale, rewards)
  return rewards
