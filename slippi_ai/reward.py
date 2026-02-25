"""Calculate rewards."""

import dataclasses
from typing import Sequence

import numpy as np

import melee
from slippi_ai.types import Game, Player

def is_dying(player_action: np.ndarray) -> np.ndarray:
  # See https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player_action <= 0xA

def process_deaths(
    player_action: np.ndarray,
    percent: np.ndarray | None = None,
    weight_by_percent: bool = False,
) -> np.ndarray:
  """Detect death frames and optionally scale them by percent taken."""

  actions = np.asarray(player_action)
  deaths = is_dying(actions)

  # Players are in a dead action state for many consecutive frames; only keep
  # the rising edge where death first registers.
  death_frames = np.diff(deaths.astype(np.int8), axis=0) > 0
  death_frames = death_frames.astype(np.float32)

  if weight_by_percent and percent is not None:
    percents = np.asarray(percent, dtype=np.float32)
    weights = 100.0 / (100.0 + percents[1:])
    death_frames *= weights

  return death_frames


def process_deaths_teams(player_action: np.ndarray, percent: np.ndarray) -> np.ndarray:
  # Backwards-compatible wrapper used by older call sites.
  return process_deaths(player_action, percent, weight_by_percent=True)

def process_damages(damages: np.ndarray) -> np.ndarray:
  damages = np.asarray(damages, dtype=np.float32)
  return np.maximum(np.diff(damages, axis=0), 0.0)

def grabbed_ledge(player_action: np.ndarray) -> np.ndarray:
  is_ledge_grab = player_action == melee.Action.EDGE_CATCHING.value
  return np.logical_and(np.logical_not(is_ledge_grab[:-1]), is_ledge_grab[1:])

def _get_bad_ledge_grabs_multi(
    player: Player, opponents: Sequence[Player]) -> np.ndarray:
  ledge_grabs = grabbed_ledge(player.action)

  if not opponents:
    return np.zeros_like(ledge_grabs, dtype=bool)

  center_direction = player.x < 0
  opponent_directions = [player.x < opponent.x for opponent in opponents]
  opponents_towards_center = np.logical_and.reduce(
      [op_dir == center_direction for op_dir in opponent_directions])
  bad_ledge_grabs = np.logical_and(ledge_grabs, opponents_towards_center[:-1])

  invulnerable = np.zeros_like(player.invulnerable, dtype=bool)
  for opponent in opponents:
    invulnerable = np.logical_or(invulnerable, opponent.invulnerable)

  bad_ledge_grabs = np.logical_and(
      bad_ledge_grabs, np.logical_not(invulnerable[:-1]))
  return bad_ledge_grabs


def get_bad_ledge_grabs(player: Player, opponent: Player) -> np.ndarray:
  return _get_bad_ledge_grabs_multi(player, [opponent])


def get_bad_ledge_grabs_team(player: Player, opponents: Sequence[Player]) -> np.ndarray:
  return _get_bad_ledge_grabs_multi(player, opponents)

def normalize(xys, epsilon=1e-6):
  r = np.sqrt(np.sum(np.square(xys), axis=-1, keepdims=True))
  return xys / (r + epsilon)

def _compute_approach_core(
    player: Player,
    opponent_xy: np.ndarray,
) -> np.ndarray:
  xy = np.stack([player.x, player.y], axis=-1)
  v = xy[1:] - xy[:-1]

  dxy = normalize(opponent_xy - xy)

  approach_factor = np.sum(v * dxy[:-1], axis=-1)

  # Player teleports when respawning.
  dying = is_dying(player.action)
  respawning = np.logical_and(dying[:-1], np.logical_not(dying[1:]))
  approach_factor = np.where(respawning, 0, approach_factor)

  return approach_factor.astype(np.float32)


def compute_approaching_factor(
    player: Player, opponent: Player) -> np.ndarray:
  """Measures how much we are approaching a single opponent on each frame."""
  opp_xy = np.stack([opponent.x, opponent.y], axis=-1)
  return _compute_approach_core(player, opp_xy)


def compute_approaching_factor_multi(
    player: Player, opponents: Sequence[Player]) -> np.ndarray:
  if not opponents:
    length = player.x.shape[0] - 1
    return np.zeros(length, dtype=np.float32)

  opponent_x = np.stack(
      [np.asarray(opponent.x, dtype=np.float32) for opponent in opponents], axis=0)
  opponent_y = np.stack(
      [np.asarray(opponent.y, dtype=np.float32) for opponent in opponents], axis=0)
  opponent_active = np.stack(
      [np.asarray(opponent.stocks_left) > 0 for opponent in opponents], axis=0)

  active_count = np.sum(opponent_active, axis=0).astype(np.float32)
  safe_count = np.maximum(active_count, 1.0)

  centroid_x = np.sum(
      np.where(opponent_active, opponent_x, 0.0), axis=0) / safe_count
  centroid_y = np.sum(
      np.where(opponent_active, opponent_y, 0.0), axis=0) / safe_count
  opp_xy = np.stack([centroid_x, centroid_y], axis=-1)

  approach_factor = _compute_approach_core(player, opp_xy)
  has_active_opponent = active_count[:-1] > 0
  approach_factor = np.where(has_active_opponent, approach_factor, 0.0)

  # In doubles, only apply approach shaping when outside the opponent x-range.
  if len(opponents) >= 2:
    inf = np.full_like(opponent_x, np.inf, dtype=np.float32)
    neg_inf = np.full_like(opponent_x, -np.inf, dtype=np.float32)
    min_x = np.min(np.where(opponent_active, opponent_x, inf), axis=0)
    max_x = np.max(np.where(opponent_active, opponent_x, neg_inf), axis=0)
    at_least_two_active = active_count >= 2
    player_x = np.asarray(player.x, dtype=np.float32)
    between = np.logical_and(player_x[:-1] >= min_x[:-1], player_x[:-1] <= max_x[:-1])
    gated = np.logical_and(at_least_two_active[:-1], between)
    approach_factor = np.where(gated, 0.0, approach_factor)

  return approach_factor.astype(np.float32)

stage_to_edge_x = {
    stage.value: x for stage, x in melee.stages.EDGE_POSITION.items()
}
get_edge_x = np.vectorize(lambda x: stage_to_edge_x.get(x, 100))

# Above this height is considered offstage for the purposes of stalling.
# The highest top platform is at 54.4 on Battlefield.
MAX_STALLING_Y = 60
DEFAULT_STALLING_THRESHOLD = 20

def amount_offstage(player: Player, stage: np.ndarray) -> np.ndarray:
  stage_xs = get_edge_x(stage[0])
  dx = np.maximum(np.abs(player.x) - stage_xs, 0)

  below = -np.minimum(player.y, 0)
  above = np.maximum(player.y - MAX_STALLING_Y, 0)
  dy = np.maximum(below, above)

  return np.sqrt(np.square(dx) + np.square(dy))

def is_stalling_offstage(
    player: Player,
    stage: np.ndarray,
    threshold: float = DEFAULT_STALLING_THRESHOLD,
) -> np.ndarray:
  return amount_offstage(player, stage) > threshold

def is_aerial_shine(player: Player):
  is_fox = player.character == melee.Character.FOX.value
  is_falco = player.character == melee.Character.FALCO.value
  is_spacie = np.logical_or(is_fox, is_falco)

  # We only care about aerial shines.
  is_shine = player.action == melee.Action.DOWN_B_AIR

  return np.logical_and(is_spacie, is_shine)

def find_offstage_shine_stalls(player: Player, stage: np.ndarray):
  return np.logical_and(
      is_stalling_offstage(player, stage),
      is_aerial_shine(player))

@dataclasses.dataclass
class RewardConfig:
  damage_ratio: float = 0.01
  ledge_grab_penalty: float = 0
  approaching_factor: float = 0
  stalling_penalty: float = 0  # per second
  stalling_threshold: float = DEFAULT_STALLING_THRESHOLD
  zelda_penalty: float = 0  # per frame
  team_size_normalization: bool = True
  singles_scale: float | None = None
  doubles_scale: float | None = None
  weight_deaths_by_percent: bool = True


def _player_has_presence(player: Player) -> bool:
  stocks = np.asarray(player.stocks_left)
  percents = np.asarray(player.percent)
  return bool(np.any(stocks > 0) or np.any(percents > 0))


def _count_active_players(players: Sequence[Player]) -> int:
  count = sum(1 for player in players if _player_has_presence(player))
  return max(count, 1)


def _add(total: np.ndarray | None, value: np.ndarray) -> np.ndarray:
  value = np.asarray(value, dtype=np.float32)
  if total is None:
    return value
  return total + value


def _stock_penalties(team: Sequence[Player]) -> np.ndarray:
  stocks = np.sum(
      [np.asarray(player.stocks_left, dtype=np.float32) for player in team],
      axis=0)
  second_last = np.where(
      (stocks[:-1] == 2) & (stocks[1:] == 1), 1.5, 0.0).astype(np.float32)
  last = np.where(
      (stocks[:-1] == 1) & (stocks[1:] == 0), 3.5, 0.0).astype(np.float32)
  return second_last + last


_ZELDA = melee.Character.ZELDA.value


def _zelda_frames(player: Player) -> np.ndarray:
  characters = np.asarray(player.character)
  grounded = np.asarray(player.on_ground, dtype=bool)
  is_grounded_zelda = np.logical_and(characters[1:] == _ZELDA, grounded[1:])
  return is_grounded_zelda.astype(np.float32)


def _team_reward(
    team: Sequence[Player],
    opponents: Sequence[Player],
    stage: np.ndarray,
    config: RewardConfig,
) -> np.ndarray:
  deaths = None
  damages = None
  ledges = None
  stalling = None
  zelda = None
  approach = None

  for player in team:
    percent = player.percent
    deaths = _add(deaths, process_deaths(
        player.action,
        percent if config.weight_deaths_by_percent else None,
        weight_by_percent=config.weight_deaths_by_percent))

    damages = _add(damages, process_damages(percent))

    ledges = _add(
        ledges,
        get_bad_ledge_grabs_team(player, opponents).astype(np.float32))

    stalling = _add(
        stalling,
        is_stalling_offstage(
            player, stage, config.stalling_threshold)[1:].astype(np.float32))

    if config.zelda_penalty:
      zelda = _add(zelda, _zelda_frames(player))

    if config.approaching_factor:
      approach = _add(
          approach, compute_approaching_factor_multi(player, opponents))

  if deaths is None:
    raise ValueError('Team must include at least one player.')

  # Normalize by the number of meaningful teammates if requested.
  normalization = (float(_count_active_players(team))
                   if config.team_size_normalization else float(len(team)))
  normalization = normalization or 1.0

  penalty = np.zeros_like(deaths, dtype=np.float32)
  penalty += deaths
  penalty += config.damage_ratio * damages
  penalty += config.ledge_grab_penalty * ledges
  penalty += (config.stalling_penalty / 60.0) * stalling

  if config.zelda_penalty and zelda is not None:
    penalty += config.zelda_penalty * zelda

  penalty /= normalization

  reward = -penalty
  reward -= _stock_penalties(team)

  if config.approaching_factor and approach is not None:
    reward += (config.approaching_factor * approach) / normalization

  return reward.astype(np.float32)


def _broadcast_is_teams(is_teams: np.ndarray, reward_shape: tuple[int, ...]) -> np.ndarray:
  mode = np.asarray(is_teams, dtype=bool)
  if mode.ndim == 0:
    return np.broadcast_to(mode, reward_shape)

  time_dim = reward_shape[0]
  if mode.shape[0] == time_dim + 1:
    mode = mode[1:]
  elif mode.shape[0] != time_dim:
    raise ValueError(
        f'is_teams first dimension {mode.shape[0]} does not align with '
        f'reward time dimension {time_dim}')

  while mode.ndim < len(reward_shape):
    mode = np.expand_dims(mode, axis=-1)

  return np.broadcast_to(mode, reward_shape)


def _apply_mode_scale(
    rewards: np.ndarray,
    is_teams: np.ndarray,
    config: RewardConfig,
) -> np.ndarray:
  singles_scale = config.singles_scale
  doubles_scale = config.doubles_scale

  if singles_scale is None and doubles_scale is None:
    return rewards

  mode = _broadcast_is_teams(is_teams, rewards.shape)

  if singles_scale is not None:
    rewards = np.where(mode, rewards, rewards * singles_scale)

  if doubles_scale is not None:
    rewards = np.where(mode, rewards * doubles_scale, rewards)

  return rewards

def compute_rewards(
    game: Game,
    damage_ratio: float = 0.01,
    ledge_grab_penalty: float = 0,
    approaching_factor: float = 0,
    stalling_penalty: float = 0,
    stalling_threshold: float = DEFAULT_STALLING_THRESHOLD,
    zelda_penalty: float = 0,
    team_size_normalization: bool = True,
    singles_scale: float | None = None,
    doubles_scale: float | None = None,
    weight_deaths_by_percent: bool = True,
) -> np.ndarray:
  """Compute per-frame rewards for the main team (ports 0/1) vs. opponents."""

  config = RewardConfig(
      damage_ratio=damage_ratio,
      ledge_grab_penalty=ledge_grab_penalty,
      approaching_factor=approaching_factor,
      stalling_penalty=stalling_penalty,
      stalling_threshold=stalling_threshold,
      zelda_penalty=zelda_penalty,
      team_size_normalization=team_size_normalization,
      singles_scale=singles_scale,
      doubles_scale=doubles_scale,
      weight_deaths_by_percent=weight_deaths_by_percent,
  )

  stage = np.asarray(game.stage)
  team1 = (game.p0, game.p1)
  team2 = (game.p2, game.p3)

  team1_reward = _team_reward(team1, team2, stage, config)
  team2_reward = _team_reward(team2, team1, stage, config)

  rewards = team1_reward - team2_reward
  rewards = _apply_mode_scale(rewards, game.is_teams, config)

  rewards = rewards.astype(np.float32)

  if not np.all(np.isfinite(rewards)):
    raise ValueError('Reward calculation produced non-finite values.')

  return rewards

def player_stats(
    player: Player,
    opponent: Player,
    stage: np.ndarray,
    stalling_threshold: float = DEFAULT_STALLING_THRESHOLD,
) -> dict:
  FPM = 60 * 60
  return dict(
      deaths=process_deaths(
          player.action, player.percent, weight_by_percent=True).mean() * FPM,
      damages=process_damages(player.percent).mean() * FPM,
      ledge_grabs=get_bad_ledge_grabs(player, opponent).mean() * FPM,
      approaching_factor=compute_approaching_factor(player, opponent).mean(),
      stalling=is_stalling_offstage(player, stage, stalling_threshold).mean(),
  )

def team_stats(team: list[Player], opponents: list[Player], stage: np.ndarray) -> dict:
  FPM = 60 * 60
  deaths = process_deaths(
      team[0].action, team[0].percent, weight_by_percent=True)
  deaths += process_deaths(
      team[1].action, team[1].percent, weight_by_percent=True)
  damages = process_damages(team[0].percent) + process_damages(team[1].percent)
  ledge_grabs = get_bad_ledge_grabs_team(team[0], opponents) + get_bad_ledge_grabs_team(team[1], opponents)

  return dict(
      deaths=deaths.mean() * FPM,
      damages=damages.mean() * FPM,
      ledge_grabs=ledge_grabs.mean() * FPM,
  )

def player_stats_from_game(game: Game, swap: bool = False) -> dict:
  p0, p1 = (game.p1, game.p0) if swap else (game.p0, game.p1)
  return player_stats(p0, p1, game.stage)

def player_team_stats_from_game(game: Game, player_index: int, teammate_index: int) -> dict:
  opponent_ports = [i for i in range(4) if i not in [player_index, teammate_index]]
  team = [getattr(game, f'p{player_index}'), getattr(game, f'p{teammate_index}')]
  opponents = [getattr(game, f'p{opponent_ports[0]}'), getattr(game, f'p{opponent_ports[1]}')]

  return team_stats(team, opponents, game.stage)

# TODO: test that the two ways of getting reward yield the same results
def get_reward(
    prev_state: melee.GameState,
    next_state: melee.GameState,
    own_port: int,
    opponent_port: int,
    damage_ratio: float = 0.01,
) -> float:
  """Reward implemented directly on gamestates."""

  def player_reward(port: int):
    players = [prev_state.players[port], next_state.players[port]]
    actions = np.array([p.action for p in players])
    percents = np.array([p.percent for p in players])
    deaths = process_deaths(
        actions, percents, weight_by_percent=True).astype(np.float32).item()

    damage = damage_ratio * process_damages(percents).item()
    return - (deaths + damage)

  return player_reward(own_port) - player_reward(opponent_port)
