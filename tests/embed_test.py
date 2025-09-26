import dataclasses
import unittest

import numpy as np
import tensorflow as tf

from slippi_ai import embed, saving, types


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


class GameEmbeddingTest(unittest.TestCase):

  def setUp(self):
    self.game = _make_game()

  def _assert_embeds(self, embedding: embed.Embedding) -> tf.Tensor:
    state = embedding.from_state(self.game)
    tensor_state = tf.nest.map_structure(tf.convert_to_tensor, state)
    embedded = embedding(tensor_state)
    self.assertEqual(embedded.shape[-1], embedding.size)
    return embedded

  def test_optional_randall_increases_size(self):
    default_embedding = embed.make_game_embedding()
    with_randall = embed.make_game_embedding(with_randall_xy=True)

    self._assert_embeds(default_embedding)
    self._assert_embeds(with_randall)

    self.assertGreater(with_randall.size, default_embedding.size)

  def test_optional_items_increases_size(self):
    default_embedding = embed.make_game_embedding()
    items_embedding = embed.make_game_embedding(
        items_config=embed.ItemsConfig(type=embed.ItemsType.MLP))

    self._assert_embeds(items_embedding)
    self.assertGreater(items_embedding.size, default_embedding.size)

  def test_optional_nana_increases_size(self):
    default_embedding = embed.make_game_embedding()
    player_cfg = dataclasses.asdict(embed.PlayerConfig(
        with_nana=True,
        legacy_jumps_left=False,
    ))
    nana_embedding = embed.make_game_embedding(player_config=player_cfg)

    self._assert_embeds(nana_embedding)
    self.assertGreater(nana_embedding.size, default_embedding.size)


class ConfigUpgradeTest(unittest.TestCase):

  def test_upgrade_adds_optional_embed_flags(self):
    old_embed = dict(
        player=dict(
            xy_scale=0.05,
            shield_scale=0.01,
            speed_scale=0.5,
            with_speeds=False,
            with_controller=False,
        ),
        controller=dict(
            axis_spacing=16,
            shoulder_spacing=4,
        ),
    )

    config = dict(version=3, embed=old_embed)
    upgraded = saving.upgrade_config(config)

    self.assertEqual(upgraded['version'], saving.VERSION)
    self.assertIn('with_randall_xy', upgraded['embed'])
    self.assertFalse(upgraded['embed']['with_randall_xy'])
    self.assertIn('items', upgraded['embed'])
    self.assertEqual(
        upgraded['embed']['items']['type'],
        embed.ItemsType.SKIP,
    )
    self.assertIn('with_nana', upgraded['embed']['player'])
    self.assertTrue(upgraded['embed']['player']['legacy_jumps_left'])


if __name__ == '__main__':
  unittest.main()
