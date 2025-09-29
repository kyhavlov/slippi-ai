"""Utility helpers for RL configuration plumbing."""

from __future__ import annotations

import math

import numpy as np


def compute_singles_mask(num_envs: int, singles_fraction: float) -> np.ndarray:
  """Return a boolean mask indicating which env indices should run singles."""
  if num_envs < 0:
    raise ValueError('num_envs must be non-negative')
  if not 0.0 <= singles_fraction <= 1.0:
    raise ValueError('singles_fraction must be in [0.0, 1.0]')
  singles_envs = int(math.floor(num_envs * singles_fraction + 0.5))
  singles_envs = min(max(singles_envs, 0), num_envs)
  mask = np.zeros(num_envs, dtype=bool)
  mask[:singles_envs] = True
  return mask


__all__ = ['compute_singles_mask']
