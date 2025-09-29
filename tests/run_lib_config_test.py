import unittest

import numpy as np

from slippi_ai.rl import config_utils


class ComputeSinglesMaskTest(unittest.TestCase):

  def test_zero_envs(self):
    mask = config_utils.compute_singles_mask(0, 0.5)
    self.assertEqual(mask.size, 0)

  def test_half_split_even(self):
    mask = config_utils.compute_singles_mask(4, 0.5)
    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))

  def test_rounds_to_nearest(self):
    mask = config_utils.compute_singles_mask(5, 0.5)
    self.assertEqual(mask.sum(), 3)

  def test_invalid_fraction_raises(self):
    with self.assertRaises(ValueError):
      config_utils.compute_singles_mask(4, 1.2)

  def test_negative_envs_raises(self):
    with self.assertRaises(ValueError):
      config_utils.compute_singles_mask(-1, 0.5)


if __name__ == '__main__':
  unittest.main(failfast=True)
