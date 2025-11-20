import pathlib
import unittest

import numpy as np

import melee

from slippi_ai import reward
from slippi_ai.types import (
    Buttons,
    Controller,
    Game,
    Item,
    Items,
    MAX_ITEMS,
    Nana,
    Player,
    Randall,
    Stick,
    game_array_to_nt,
)

try:  # pragma: no cover - import guard mirrors other tests
  from slippi_db import parse_peppi  # type: ignore
except ImportError:  # pragma: no cover
  parse_peppi = None


def _zeros_bool(length: int) -> np.ndarray:
  return np.zeros(length, dtype=np.bool_)


def _zeros_uint8(length: int) -> np.ndarray:
  return np.zeros(length, dtype=np.uint8)


def _zeros_uint16(length: int) -> np.ndarray:
  return np.zeros(length, dtype=np.uint16)


def _zeros_float32(length: int) -> np.ndarray:
  return np.zeros(length, dtype=np.float32)


def make_buttons(length: int) -> Buttons:
  return Buttons(*[_zeros_bool(length) for _ in Buttons._fields])


def make_controller(length: int) -> Controller:
  stick = Stick(x=_zeros_float32(length), y=_zeros_float32(length))
  return Controller(
      main_stick=stick,
      c_stick=stick,
      shoulder=_zeros_float32(length),
      buttons=make_buttons(length),
  )


def make_nana(length: int) -> Nana:
  return Nana(
      exists=_zeros_bool(length),
      percent=_zeros_uint16(length),
      facing=_zeros_bool(length),
      x=_zeros_float32(length),
      y=_zeros_float32(length),
      action=_zeros_uint16(length),
      invulnerable=_zeros_bool(length),
      character=_zeros_uint8(length),
      jumps_left=_zeros_uint8(length),
      shield_strength=_zeros_float32(length),
      on_ground=_zeros_bool(length),
  )


def make_items(length: int) -> Items:
  def empty_item() -> Item:
    return Item(
        exists=_zeros_bool(length),
        type=_zeros_uint16(length),
        state=_zeros_uint8(length),
        x=_zeros_float32(length),
        y=_zeros_float32(length),
    )

  return Items(*[empty_item() for _ in range(MAX_ITEMS)])


def make_player(
    percent: list[int],
    action: list[int],
    stocks_left: list[int],
    *,
    character: melee.Character = melee.Character.FOX,
    character_sequence: list[melee.Character] | None = None,
    x: list[float] | None = None,
    y: list[float] | None = None,
    invulnerable: list[bool] | None = None,
    is_dead: list[bool] | None = None,
    on_ground: list[bool] | None = None,
) -> Player:
  percent_arr = np.asarray(percent, dtype=np.uint16)
  length = percent_arr.shape[0]
  if character_sequence is not None:
    characters = np.asarray([c.value for c in character_sequence], dtype=np.uint8)
  else:
    characters = np.full(length, character.value, dtype=np.uint8)

  return Player(
      percent=percent_arr,
      facing=_zeros_bool(length),
      x=np.asarray(x if x is not None else [0.0] * length, dtype=np.float32),
      y=np.asarray(y if y is not None else [0.0] * length, dtype=np.float32),
      action=np.asarray(action, dtype=np.uint16),
      invulnerable=np.asarray(
          invulnerable if invulnerable is not None else [False] * length,
          dtype=np.bool_,
      ),
      character=characters,
      jumps_left=_zeros_uint8(length),
      shield_strength=_zeros_float32(length),
      on_ground=np.asarray(
          on_ground if on_ground is not None else [False] * length, dtype=np.bool_),
      is_dead=np.asarray(
          is_dead if is_dead is not None else [False] * length, dtype=np.bool_),
      stocks_left=np.asarray(stocks_left, dtype=np.uint8),
      controller=make_controller(length),
      nana=make_nana(length),
  )


def make_game(
    team1: tuple[Player, Player],
    team2: tuple[Player, Player],
    *,
    is_teams: bool,
) -> Game:
  length = team1[0].percent.shape[0]
  stage = np.full(length, melee.Stage.BATTLEFIELD.value, dtype=np.uint8)
  randall_phase = _zeros_float32(length)
  randall = Randall(
      x=_zeros_float32(length),
      y=_zeros_float32(length),
  )
  items = make_items(length)
  return Game(
      p0=team1[0],
      p1=team1[1],
      p2=team2[0],
      p3=team2[1],
      stage=stage,
      randall_phase=randall_phase,
      randall=randall,
      items=items,
      is_teams=np.full(length, is_teams, dtype=np.bool_),
  )


def make_placeholder(length: int) -> Player:
  return make_player(
      percent=[0] * length,
      action=[0xE] * length,
      stocks_left=[0] * length,
      is_dead=[True] * length,
  )


class RewardTest(unittest.TestCase):

  def test_singles_damage_not_halved(self):
    length = 3
    p0 = make_player(
        percent=[0, 10, 20],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    rewards_default = reward.compute_rewards(game)
    rewards_disabled = reward.compute_rewards(
        game, team_size_normalization=False)

    # First timestep only has damage; ensure the normalized reward is twice
    # the legacy value that divided by team size 2 regardless of mode.
    self.assertAlmostEqual(rewards_default[0], -0.1, places=5)
    self.assertAlmostEqual(rewards_disabled[0], -0.05, places=5)

  def test_doubles_damage_matches_legacy(self):
    length = 3
    p0 = make_player(
        percent=[0, 10, 10],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p1 = make_player(
        percent=[0, 10, 10],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )

    game = make_game((p0, p1), (p2, p3), is_teams=True)

    rewards_default = reward.compute_rewards(game)
    rewards_disabled = reward.compute_rewards(
        game, team_size_normalization=False)

    np.testing.assert_allclose(rewards_default, rewards_disabled)

  def test_ledge_grab_penalty_applies(self):
    length = 3
    p0 = make_player(
        percent=[0, 0, 0],
        action=[0, melee.Action.EDGE_CATCHING.value, melee.Action.EDGE_CATCHING.value],
        stocks_left=[4, 4, 4],
        x=[-10.0, -10.0, -10.0],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        x=[-5.0, -5.0, -5.0],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    baseline = reward.compute_rewards(game)
    penalized = reward.compute_rewards(game, ledge_grab_penalty=0.5)

    self.assertAlmostEqual(baseline[0], 0.0, places=6)
    self.assertAlmostEqual(penalized[0], -0.5, places=6)
    self.assertAlmostEqual(penalized[1], baseline[1], places=6)

  def test_stalling_penalty_applies(self):
    length = 3
    p0 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        y=[0.0, 90.0, 0.0],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    baseline = reward.compute_rewards(game)
    penalized = reward.compute_rewards(game, stalling_penalty=0.6)

    expected = -0.6 / 60.0
    self.assertAlmostEqual(penalized[0], expected, places=6)
    self.assertAlmostEqual(penalized[1], baseline[1], places=6)

  def test_approaching_factor_reward(self):
    length = 3
    p0 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        x=[-30.0, -20.0, -10.0],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        x=[0.0, 0.0, 0.0],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    baseline = reward.compute_rewards(game)
    rewarded = reward.compute_rewards(game, approaching_factor=0.01)

    self.assertGreater(rewarded[0], baseline[0])
    self.assertGreater(rewarded[1], baseline[1])
    np.testing.assert_allclose(rewarded, baseline + 0.1, rtol=1e-5)

  def test_mode_scaling(self):
    length = 3
    singles = make_game(
        (
            make_player(percent=[0, 10, 20], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
            make_placeholder(length),
        ),
        (
            make_player(percent=[0, 0, 0], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
            make_placeholder(length),
        ),
        is_teams=False,
    )
    doubles = make_game(
        (
            make_player(percent=[0, 10, 10], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
            make_player(percent=[0, 0, 0], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
        ),
        (
            make_player(percent=[0, 0, 0], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
            make_player(percent=[0, 0, 0], action=[0xE, 0xE, 0xE], stocks_left=[4, 4, 4]),
        ),
        is_teams=True,
    )

    singles_base = reward.compute_rewards(singles)
    singles_scaled = reward.compute_rewards(singles, singles_scale=3.0)
    np.testing.assert_allclose(singles_scaled, singles_base * 3.0)

    doubles_base = reward.compute_rewards(doubles)
    doubles_scaled = reward.compute_rewards(doubles, doubles_scale=0.5)
    np.testing.assert_allclose(doubles_scaled, doubles_base * 0.5)

  def test_zelda_penalty_applied_per_frame(self):
    length = 3
    p0 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        character_sequence=[
            melee.Character.SHEIK,
            melee.Character.ZELDA,
            melee.Character.SHEIK,
        ],
        on_ground=[False, True, False],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    baseline = reward.compute_rewards(game)
    penalized = reward.compute_rewards(game, zelda_penalty=0.002)

    self.assertAlmostEqual(penalized[0], baseline[0] - 0.002, places=6)
    self.assertAlmostEqual(penalized[1], baseline[1], places=6)

  def test_zelda_penalty_requires_grounded(self):
    length = 3
    p0 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
        character_sequence=[
            melee.Character.SHEIK,
            melee.Character.ZELDA,
            melee.Character.SHEIK,
        ],
        on_ground=[False, False, False],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    baseline = reward.compute_rewards(game)
    penalized = reward.compute_rewards(game, zelda_penalty=0.002)

    np.testing.assert_allclose(penalized, baseline)

  def test_death_weighting_toggle(self):
    length = 3
    p0 = make_player(
        percent=[0, 200, 200],
        action=[0xE, 0xE, 0x0],
        stocks_left=[4, 4, 4],
    )
    p1 = make_placeholder(length)
    p2 = make_player(
        percent=[0, 0, 0],
        action=[0xE, 0xE, 0xE],
        stocks_left=[4, 4, 4],
    )
    p3 = make_placeholder(length)

    game = make_game((p0, p1), (p2, p3), is_teams=False)

    weighted = reward.compute_rewards(game)
    unweighted = reward.compute_rewards(game, weight_deaths_by_percent=False)

    self.assertAlmostEqual(weighted[1], -100 / 300, places=5)
    self.assertAlmostEqual(unweighted[1], -1.0, places=5)


@unittest.skipUnless(parse_peppi is not None, 'peppi_py is required for replay-based tests')
class RewardReplayTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._replay_dir = pathlib.Path(__file__).parent / 'data' / 'replays'

  def _sum_reward_for(self, game: Game, order: tuple[int, int, int, int]) -> float:
    reordered = Game(
        p0=getattr(game, f'p{order[0]}'),
        p1=getattr(game, f'p{order[1]}'),
        p2=getattr(game, f'p{order[2]}'),
        p3=getattr(game, f'p{order[3]}'),
        stage=game.stage,
        randall_phase=game.randall_phase,
        randall=game.randall,
        items=game.items,
        is_teams=game.is_teams,
    )
    rewards = reward.compute_rewards(reordered)
    return float(np.sum(rewards))

  def _load_game(self, filename: str) -> Game:
    path = self._replay_dir / filename
    game_struct = parse_peppi.get_slp(str(path))
    return game_array_to_nt(game_struct)

  def test_real_singles_replay_rewards(self):
    game = self._load_game('test_singles_game.slp')
    rewards = reward.compute_rewards(game)

    self.assertEqual(rewards.shape[0], game.p0.percent.shape[0] - 1)
    self.assertTrue(np.all(np.isfinite(rewards)))
    self.assertTrue(np.any(rewards != 0))

    scaled = reward.compute_rewards(game, singles_scale=2.0)
    np.testing.assert_allclose(scaled, rewards * 2.0)

    team_main = self._sum_reward_for(game, (0, 1, 2, 3))
    team_placeholder = self._sum_reward_for(game, (1, 0, 2, 3))
    opponent = self._sum_reward_for(game, (2, 3, 0, 1))
    opponent_placeholder = self._sum_reward_for(game, (3, 2, 0, 1))

    self.assertAlmostEqual(team_main, team_placeholder, places=5)
    self.assertAlmostEqual(opponent, opponent_placeholder, places=5)
    self.assertAlmostEqual(team_main, -opponent, places=5)
    self.assertAlmostEqual(team_main, -4.254157066345215, places=5)
    self.assertAlmostEqual(opponent, 4.254157066345215, places=5)

  def test_real_doubles_replay_rewards(self):
    game = self._load_game('test_doubles_game.slp')
    rewards = reward.compute_rewards(game)

    self.assertEqual(rewards.shape[0], game.p0.percent.shape[0] - 1)
    self.assertTrue(np.all(np.isfinite(rewards)))
    self.assertTrue(np.any(rewards != 0))

    scaled = reward.compute_rewards(game, doubles_scale=0.25)
    np.testing.assert_allclose(scaled, rewards * 0.25)

    team_a = self._sum_reward_for(game, (0, 3, 1, 2))
    team_a_teammate = self._sum_reward_for(game, (3, 0, 1, 2))
    team_b = self._sum_reward_for(game, (1, 2, 0, 3))
    team_b_teammate = self._sum_reward_for(game, (2, 1, 0, 3))

    self.assertAlmostEqual(team_a, team_a_teammate, places=5)
    self.assertAlmostEqual(team_b, team_b_teammate, places=5)
    self.assertAlmostEqual(team_a, -team_b, places=5)
    self.assertAlmostEqual(team_a, -8.381505966186523, places=5)
    self.assertAlmostEqual(team_b, 8.381505966186523, places=5)


if __name__ == '__main__':
  unittest.main()
