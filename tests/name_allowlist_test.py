import unittest

from melee import Character

from slippi_ai import evaluators
from slippi_ai.rl import run_lib


class NameAllowlistParseTest(unittest.TestCase):

  def test_parses_basic_spec_with_fallback(self):
    result = run_lib._parse_name_allowlist('ALL:Master Player,Fox:Cody,Cactuar')
    self.assertEqual(result.fallback, ['Master Player'])
    self.assertIn(Character.FOX, result.per_character)
    self.assertEqual(result.per_character[Character.FOX], ['Cody', 'Cactuar'])

  def test_deduplicates_entries(self):
    spec = 'Fox:Cody,IBDW, Cody ;ALL:Master Player , Master Player'
    result = run_lib._parse_name_allowlist(spec)
    self.assertEqual(result.per_character[Character.FOX], ['Cody', 'IBDW'])
    self.assertEqual(result.fallback, ['Master Player'])

  def test_unknown_character_raises(self):
    with self.assertRaises(ValueError):
      run_lib._parse_name_allowlist('UnknownChar:PlayerOne')

  def test_fallback_names_extend_every_character(self):
    parsed = run_lib._parse_name_allowlist(
        'ALL:Master Player,Diamond Player,Fox:Cody')
    config = evaluators.NameSelectionConfig(
        per_character=parsed.per_character,
        fallback=parsed.fallback,
    )

    fox_pool = evaluators._name_pool_for_character(config, Character.FOX)
    self.assertEqual(fox_pool, ['Cody', 'Master Player', 'Diamond Player'])

    falco_pool = evaluators._name_pool_for_character(config, Character.FALCO)
    self.assertEqual(falco_pool, ['Master Player', 'Diamond Player'])


if __name__ == '__main__':
  unittest.main()
