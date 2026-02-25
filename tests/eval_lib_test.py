import unittest

from slippi_ai import eval_lib


class EvalLibTest(unittest.TestCase):

  def test_get_name_code_empty_map_defaults_to_zero(self):
    state = dict(name_map={})
    self.assertEqual(eval_lib.get_name_code(state, 'anything'), 0)

  def test_get_name_code_missing_name_map_defaults_to_zero(self):
    state = {}
    self.assertEqual(eval_lib.get_name_code(state, 'anything'), 0)

  def test_get_name_code_exact_match(self):
    state = dict(name_map={'Master Player': 3})
    self.assertEqual(eval_lib.get_name_code(state, 'Master Player'), 3)

  def test_get_name_code_normalized_match(self):
    state = dict(name_map={'Mang0': 7})
    self.assertEqual(eval_lib.get_name_code(state, 'mang'), 7)

  def test_get_name_code_unknown_raises_for_nonempty_map(self):
    state = dict(name_map={'Master Player': 0})
    with self.assertRaises(ValueError):
      eval_lib.get_name_code(state, 'Unknown')

  def test_get_name_from_rl_state_empty_string_returns_none(self):
    state = dict(rl_config=dict(agent=dict(name='')))
    self.assertIsNone(eval_lib.get_name_from_rl_state(state))

  def test_get_name_from_rl_state_empty_list_returns_none(self):
    state = dict(rl_config=dict(agent=dict(name=[])))
    self.assertIsNone(eval_lib.get_name_from_rl_state(state))

  def test_get_name_from_rl_state_filters_blank_entries(self):
    state = dict(rl_config=dict(agent=dict(name=['', '  ', 'Master Player'])))
    self.assertEqual(eval_lib.get_name_from_rl_state(state), ['Master Player'])


if __name__ == '__main__':
  unittest.main(failfast=True)
