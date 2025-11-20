# Singles/Doubles RL Bug Snapshot (2025-11-16)

Update (2025-11-20): The broken mixed-mode path has been removed; the RL pipeline is doubles-only until a new singles implementation lands. Keeping the notes below for historical context.

While integrating mixed singles+doubles rollouts we never finished splitting 1v1 trajectories apart before they reach the learner. `_SinglesChunk.recv` currently merges two independent singles matches into a single `EnvOutput` with ports 1–4, and `LearnerManager` batches that as if it were a 2v2 game. In `Learner.ppo` (and in logging) we still assume `p0/p1` are teammates fighting `p2/p3`, so in singles half of the batch gets inverted rewards/advantages and pushes the policy away from the teacher immediately. Any run with `enable_singles=True` will therefore see high forward KL and no improvement until we split those trajectories and pass mode-aware reward masks.

Fix (later): track singles chunks when dequeuing env outputs, emit two-port trajectories for each 1v1 game, and plumb a mode flag so reward/logging code uses the correct opponent mapping.
