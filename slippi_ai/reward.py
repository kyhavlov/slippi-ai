"""Calculate rewards."""

import dataclasses

import numpy as np

import melee
from slippi_ai.types import Game, Player

def is_dying(player_action: np.ndarray) -> np.ndarray:
  # See https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player_action <= 0xA

def process_deaths(player_action: np.ndarray, percent: np.ndarray) -> np.ndarray:
  deaths = is_dying(player_action)
  # Players are in a dead action-state for many consecutive frames.
  # Prune all but the first frame of death
  death_frames = np.logical_and(np.logical_not(deaths[:-1]), deaths[1:])

  # Reduce the value of the death frame by 100/(100+percent)
  # This way, a death at 0 percent is worth the full amount, but a death at
  # 100 percent is worth half, and a death at 200 percent is worth a third, and so on.
  #clipped_percent = np.clip(percent, 0, 999)[1:]
  return (100 / (100 + percent[1:])) * death_frames

def process_damages(damages: np.ndarray) -> np.ndarray:
  damages = damages.astype(np.float32)
  return np.maximum(damages[1:] - damages[:-1], 0)

def grabbed_ledge(player_action: np.ndarray) -> np.ndarray:
  is_ledge_grab = player_action == melee.Action.EDGE_CATCHING.value
  return np.logical_and(np.logical_not(is_ledge_grab[:-1]), is_ledge_grab[1:])

def get_bad_ledge_grabs(player: Player, opponents: list[Player]) -> np.ndarray:
  ledge_grabs = grabbed_ledge(player.action)

  # Don't penalize if opponent is offstage
  opponent_direction = np.logical_and(player.x < opponents[0].x, player.x < opponents[1].x)  # True if opponents are right
  center_direction = player.x < 0  # True if center is right
  opponent_towards_center = opponent_direction == center_direction
  bad_ledge_grabs = np.logical_and(ledge_grabs, opponent_towards_center[:-1])

  # Also ok if opponent is invincible (like after respawn)
  bad_ledge_grabs = np.logical_and(
      bad_ledge_grabs, np.logical_not(opponents.invulnerable[:-1]))

  return bad_ledge_grabs

def get_bad_ledge_grabs_team(player: Player, opponents: list[Player]) -> np.ndarray:
  ledge_grabs = grabbed_ledge(player.action)

  # Don't penalize if opponent is offstage
  opponent1_direction = player.x < opponents[0].x  # True if opponent is right
  opponent2_direction = player.x < opponents[1].x  # True if opponent is right
  center_direction = player.x < 0  # True if center is right
  opponents_towards_center = np.logical_and(opponent1_direction == center_direction, opponent2_direction == center_direction)
  bad_ledge_grabs = np.logical_and(ledge_grabs, opponents_towards_center[:-1])

  # Also ok if opponent is invincible (like after respawn)
  bad_ledge_grabs = np.logical_and(
      bad_ledge_grabs, np.logical_not(np.logical_or(opponents[0].invulnerable[:-1], opponents[1].invulnerable[:-1])))

  return bad_ledge_grabs

def normalize(xys, epsilon=1e-6):
  r = np.sqrt(np.sum(np.square(xys), axis=-1, keepdims=True))
  return xys / (r + epsilon)

def compute_approaching_factor(
    player: Player, opponent: Player) -> np.ndarray:
  """Measures how much we are approaching the opponent on each frame."""
  xy = np.stack([player.x, player.y], axis=-1)
  v = xy[1:] - xy[:-1]


  opp_xy = np.stack([opponent.x, opponent.y], axis=-1)
  dxy = normalize(opp_xy - xy)

  approach_factor = np.sum(v * dxy[:-1], axis=-1)

  # Player teleports when respawning.
  dying = is_dying(player.action)
  respawning = np.logical_and(dying[:-1], np.logical_not(dying[1:]))
  approach_factor = np.where(respawning, 0, approach_factor)

  return approach_factor

stage_to_edge_x = {
    stage.value: x for stage, x in melee.stages.EDGE_POSITION.items()
}
get_edge_x = np.vectorize(lambda x: stage_to_edge_x.get(x, 100))

# Above this height is considered offstage for the purposes of stalling.
# The highest top platform is at 54.4 on Battlefield.
MAX_STALLING_Y = 60

def amount_offstage(player: Player, stage: np.ndarray) -> np.ndarray:
  stage_xs = get_edge_x(stage[0])
  dx = np.maximum(np.abs(player.x) - stage_xs, 0)

  below = -np.minimum(player.y, 0)
  above = np.maximum(player.y - MAX_STALLING_Y, 0)
  dy = np.maximum(below, above)

  return np.sqrt(np.square(dx) + np.square(dy))

def is_stalling_offstage(player: Player, stage: np.ndarray) -> np.ndarray:
  return amount_offstage(player, stage) > 20  # arbitrary

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

def compute_rewards(
    game: Game,
    damage_ratio: float = 0.01,
    ledge_grab_penalty: float = 0,
    approaching_factor: float = 0,
    stalling_penalty: float = 0  # per second
) -> np.ndarray:
  '''
    Args:
      game: nest of np.arrays of length T, from make_dataset.py
      damage_ratio: How much damage (percent) counts relative to stocks
    Returns:
      A length (T-1) np.array of rewards
  '''

  '''def player_reward(player: Player, opponent: Player):
    deaths = process_deaths(player.action).astype(np.float32)
    damages = damage_ratio * process_damages(player.percent)

    bad_ledge_grabs = get_bad_ledge_grabs(player, opponent).astype(np.float32)
    ledge_grab_penalties = ledge_grab_penalty * bad_ledge_grabs

    stalling = is_stalling_offstage(player, game.stage)[1:]
    stalling_penalties = (stalling_penalty / 60) * stalling.astype(np.float32)

    # ignore approaching factor for now in doubles
    #reward = approaching_factor * compute_approaching_factor(player, opponent)
    reward = (deaths + damages + ledge_grab_penalties + stalling_penalties)

    return reward'''
  
  # calculate rewards for a team
  def team_reward(team: list[Player], opponents: list[Player]):
    deaths = process_deaths(team[0].action, team[0].percent).astype(np.float32)
    deaths += process_deaths(team[1].action, team[1].percent).astype(np.float32)
    damages = damage_ratio * process_damages(team[0].percent)
    damages += damage_ratio * process_damages(team[1].percent)

    bad_ledge_grabs = get_bad_ledge_grabs_team(team[0], opponents).astype(np.float32)
    bad_ledge_grabs += get_bad_ledge_grabs_team(team[1], opponents).astype(np.float32)
    ledge_grab_penalties = ledge_grab_penalty * bad_ledge_grabs

    '''stalling = is_stalling_offstage(team[0], game.stage)[1:]
    stalling += is_stalling_offstage(team[1], game.stage)[1:]
    stalling_penalties = (stalling_penalty / 60) * stalling.astype(np.float32)'''

    # ignore ledge_grab_penalties and stalling_penalties for now in doubles
    return -(deaths + damages + ledge_grab_penalties)/2.0

  # Zero-sum rewards ensure there can be no collusion.
  # rewards = player_reward(game.p0, game.p1) - player_reward(game.p1, game.p0)

  # separate players into teams so we can calculate reward from the
  # main player/team1's perspective
  team1 = [game.p0, game.p1]
  team2 = [game.p2, game.p3]

  rewards = team_reward(team1, team2) - team_reward(team2, team1)

  # sanity checks
  assert np.all(rewards > -2)
  assert np.all(rewards < 2)
  assert rewards.dtype == np.float32

  return rewards

def player_stats(player: Player, opponent: Player, stage: np.ndarray) -> dict:
  FPM = 60 * 60
  return dict(
      deaths=process_deaths(player.action, player.percent).mean() * FPM,
      damages=process_damages(player.percent).mean() * FPM,
      ledge_grabs=get_bad_ledge_grabs(player, opponent).mean() * FPM,
      approaching_factor=compute_approaching_factor(player, opponent).mean(),
      stalling=is_stalling_offstage(player, stage).mean(),
  )

def team_stats(team: list[Player], opponents: list[Player], stage: np.ndarray) -> dict:
  FPM = 60 * 60
  deaths = process_deaths(team[0].action, team[0].percent) + process_deaths(team[1].action, team[1].percent)
  damages = process_damages(team[0].percent) + process_damages(team[1].percent)
  ledge_grabs = get_bad_ledge_grabs_team(team[0], opponents) + get_bad_ledge_grabs_team(team[1], opponents)
  stalling = is_stalling_offstage(team[0], stage) + is_stalling_offstage(team[1], stage)

  return dict(
      deaths=deaths.mean() * FPM,
      damages=damages.mean() * FPM,
      ledge_grabs=ledge_grabs.mean() * FPM,
      stalling=stalling.mean(),
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
    deaths = process_deaths(actions, percents).astype(np.float32).item()

    damage = damage_ratio * process_damages(percents).item()
    return - (deaths + damage)

  return player_reward(own_port) - player_reward(opponent_port)
