import os
import tempfile
import unittest

import melee

from slippi_ai import dolphin
from slippi_ai import instant_match


class _DummyConsole:

  def __init__(self, home_path: str):
    self._home_path = home_path

  def _get_dolphin_home_path(self) -> str:
    return self._home_path


class InstantMatchTest(unittest.TestCase):

  def test_resolve_config_derives_pools(self):
    players = {
        1: dolphin.AI(character_weight_table={
            melee.Character.FOX: 3,
            melee.Character.FALCO: 1,
        }),
        2: dolphin.AI(character=melee.Character.MARTH),
    }

    config = instant_match.resolve_config(
        players=players,
        stage=melee.Stage.RANDOM_STAGE,
        starting_stocks=1,
    )

    self.assertEqual(
        config.character_pool,
        (melee.Character.FOX, melee.Character.FALCO, melee.Character.MARTH),
    )
    self.assertEqual(
        config.stage_pool,
        (
            melee.Stage.BATTLEFIELD,
            melee.Stage.FINAL_DESTINATION,
            melee.Stage.DREAMLAND,
            melee.Stage.POKEMON_STADIUM,
            melee.Stage.YOSHIS_STORY,
            melee.Stage.FOUNTAIN_OF_DREAMS,
        ),
    )
    self.assertEqual(config.starting_stocks, 1)

  def test_inject_gecko_codes_updates_ini(self):
    config = instant_match.InstantMatchConfig(
        character_pool=(melee.Character.FOX, melee.Character.FALCO),
        stage_pool=(melee.Stage.BATTLEFIELD, melee.Stage.FINAL_DESTINATION),
        starting_stocks=1,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      game_settings = os.path.join(temp_dir, 'GameSettings')
      os.makedirs(game_settings, exist_ok=True)
      ini_path = os.path.join(game_settings, 'GALE01r2.ini')
      with open(ini_path, 'w') as f:
        f.write('[Gecko_Enabled]\n[Gecko]\n')

      instant_match.inject_gecko_codes(_DummyConsole(temp_dir), config)

      with open(ini_path) as f:
        contents = f.read()

    self.assertIn('$slippi-ai: Instant Match Randomizer', contents)
    self.assertIn('C21A5C14 00000001', contents)
    self.assertIn('C21A5C20 00000001', contents)
    self.assertIn('38600000 60000000', contents)
    self.assertEqual(contents.count('C21A5C14'), 1)
    self.assertEqual(contents.count('C21A5C20'), 1)
    self.assertNotIn('C21B15A0', contents)

  def test_inject_gecko_codes_replaces_existing_conflicting_hooks(self):
    config = instant_match.InstantMatchConfig(
        character_pool=(melee.Character.FOX,),
        stage_pool=(melee.Stage.BATTLEFIELD,),
        starting_stocks=1,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      game_settings = os.path.join(temp_dir, 'GameSettings')
      os.makedirs(game_settings, exist_ok=True)
      ini_path = os.path.join(game_settings, 'GALE01r2.ini')
      with open(ini_path, 'w') as f:
        f.write(
            '[Gecko_Enabled]\n'
            '[Gecko]\n'
            '$Old Rematch Hook\n'
            'C21A5C14 00000001\n'
            '60000000 60000000\n'
            '$Old Sudden Death Hook\n'
            'C21A5C20 00000001\n'
            '60000000 60000000\n'
            '$Old Start Hook\n'
            'C21B15A0 00000001\n'
            '60000000 60000000\n'
            '$Old Reenter Hook\n'
            'C21A5E90 00000001\n'
            '60000000 60000000\n')

      instant_match.inject_gecko_codes(_DummyConsole(temp_dir), config)

      with open(ini_path) as f:
        contents = f.read()

    self.assertNotIn('$Old Rematch Hook', contents)
    self.assertNotIn('$Old Sudden Death Hook', contents)
    self.assertNotIn('$Old Start Hook', contents)
    self.assertNotIn('$Old Reenter Hook', contents)
    self.assertIn('$slippi-ai: Instant Match Randomizer', contents)
    self.assertEqual(contents.count('C21A5C14'), 1)
    self.assertEqual(contents.count('C21A5C20'), 1)
    self.assertNotIn('C21B15A0', contents)

  def test_build_gecko_code_is_static_and_toolchain_free(self):
    code = instant_match._build_gecko_code()

    self.assertIn('C21A5C14 00000001', code)
    self.assertIn('C21A5C20 00000001', code)
    self.assertEqual(code.count('38600000 60000000'), 2)


if __name__ == '__main__':
  unittest.main()
