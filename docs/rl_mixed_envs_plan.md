# Mixed Singles + Doubles RL: Design and Implementation Plan

Author: LLM planning pass on 2025-09-26

Status: Planning only (no code changes yet)

Scope: Enable robust RL training with a controllable mix of singles and doubles environments, targeting a default 50/50 trajectory split. Remove fragile multi-Dolphin-per-env behavior and make shapes, rewards, and logging explicitly support mixed-mode training.
Much of the core of these changes will likely like in slippi_ai/rl/run_lib.py and slippi_ai/envs.py.

---

## Goals

- Train with a configurable fraction of singles vs doubles during RL (default 50/50) to improve robustness and skill balance.
- Use exactly one Dolphin process per environment for both modes to simplify resets/lifecycle and isolate failures.
- Keep the existing 4-slot `Game` schema for all environments to minimize invasive changes in the actor/evaluator pipeline.
- Prevent dummy ports from polluting gradients by masking inactive columns in the learner.
- Make reward scales comparable across modes (configurable), and add per‑mode logging to monitor stability and progress.

## Decisions

1) One-Dolphin-per-env for both singles and doubles
   - Singles envs expose a 4-slot `Game` with real players in `p0` (self) and the opponent in an even split of  either `p2` pr `p3` (to avoid bias when carrying over knowledge to doubles), `p1` and the unused opponent slot are dead placeholders.
   - `is_teams=False` for singles env states; `is_teams=True` for doubles.
   - Env ignores controllers for `p1` and the unused opponet slot in singles mode.

2) Keep 4 agents/ports at the rollout layer; add an active-column mask
   - Don’t reshape the pipeline to 2 ports for singles; instead, provide a per‑column `active_mask` so the learner trains only on live ports. Need to carefully test this both locally and end-to-end to make sure the exact correct trajectories and their shapes make it to the learner in the state we expect.
   - Effective batch columns per rollout become `4 * (#doubles_envs) + 2 * (#singles_envs)`.
   - Need to look into disabling inference for inactive singles ports, will likely require some refactoring in the actor/agent code but is worth it as inference time is our bottleneck during rollout generation currently.

3) Ratio control via `singles_fraction`
   - Replace `enable_singles` (bool) with `singles_fraction: float` on the actor config.
   - Validate realizability given `num_envs` and batching strategy (see below). Default to 0.5.

4) Reward normalization by mode
   - Figure out a reasonable default for reward settings to normalize reward in singles games if needed. Env modes can be fixed for the entire training duration, so we shouldn't need to worry about trajectory data for a game switching modes part-way through.

5) Observability
   - Fix `is_teams` values for RL envs and log per‑mode metrics: realized mix ratio, FPS/MPS, actor_kl, teacher_kl, UEV, reward mean/std, PPO objective.
   - (TODO) Metrics may need somewhat more of an overhaul to simplify and make them more useful in general, given that when I adapted this pipeline from singles to doubles initially, I didn't really make many changes to the metrics being emitted and many of them may be incorrect or not make sense. This is more of a nice-to-have/stretch goal.

## Current State (what exists now)

- Mixed mode is partially implemented only for async envs by flipping `enable_singles` on alternating inner groups. This uses two Dolphins per singles “env,” sharing one logical slot (fragile across resets and port allocation). See:
  - `slippi_ai/envs.py` AsyncBatchedEnvironmentMP env building and `enable_singles and i % 2 == 0` logic.
  - `Environment` creating two Dolphins when `enable_singles=True` (singles) and a single Dolphin when False (doubles).
  - `parse_libmelee.get_game(...)` always sets `is_teams=True` in RL path (needs correction).
- Evaluator and learner assume 4 ports and batch by concatenation; reward function already computes team-differences and works for both modes if singles are represented with placeholders.

Risks in current implementation:
- Ratio is implicit (every other inner group), not configurable beyond 50/50.
- Extra Dolphins (+50% ports) for singles; brittle port allocation; harder resets.
- Sync envs with `enable_singles=True` are not correctly provisioned.
- `is_teams` is unreliable (always true in RL path), undermining per‑mode logic.
- No per‑mode logging; hard to see skew or instability.

## Target Architecture

### Environment (one Dolphin per env)

- Singles env:
  - One Dolphin instance.
  - Observations map to a 4-slot `Game` with:
    - `p0`: self (controlled by our agent)
    - `p1`: placeholder dead player
    - `p2`/`p3`: 50/50 split randomized among opponent and placeholder dead player (can be fixed per-env, just needs to have a 50/50 balance overall)
    - `is_teams=False`
  - `step` ignores incoming controllers for `p1` and placeholder dead opponent.

- Doubles env:
  - One Dolphin instance as today.
  - `is_teams=True`.

### Ratio control & provisioning

- `singles_fraction: float` (0..1) on `ActorConfig` replaces `enable_singles`.
- Compute `singles_envs = round(num_envs * singles_fraction)`.
- Provision exactly one dolphin port per env (`num_envs` total), no extra provisioning for singles.
- Assignment policy:
  - Option A (simple): keep group-level toggling for async envs; require `singles_envs` be a multiple of `inner_batch_size`. Validate at startup; fail fast otherwise.
  - Option B (better): per‑env toggling with a boolean mask passed to the env builder; removes divisibility constraints, but requires minor plumbing to deliver a per‑env mode flag.
- (IMPORTANT) Need to keep in mind that the current code has one agent per game port (1-4) that performs batch inference for that port across all envs. We will need to come up with a solution to change this to handle the fact that each singles env will only want 2 ports worth of inference/agents, not 4.

### Rollout packaging & learner masks

- Continue to build 4-port trajectories for every env as today.
- Add `active_mask: np.ndarray[bool]` to `evaluators.Trajectory` with shape `[B]` (one boolean per batch column):
  - For doubles columns (p0,p1,p2,p3) → True.
  - For singles columns → True for p0 and active opponent slot, False for p1, and inactive opponent slot.
- Learner changes:
  - When computing PPO, first gather/slice all frame tensors, actions, advantages, and is_resetting using `active_mask` along the batch dimension.
  - Hidden-state sizing: compute the effective batch size at init as the sum of active columns from the dummy trajectory for the configured ratio (or infer per rollout and handle via masked slicing).
  - Maintain variable shapes otherwise; this is a selection (mask) operation, not a reshape.

### Rewards and normalization

- TODO: see if anything significant is still needed here or if configuring existing normalization setting is enough.

### Logging & metrics

- Record per‑mode aggregates each flush (in `rl/run_lib.py`):
  - Realized singles share, FPS/MPS, actor_kl mean/max, teacher_kl, UEV.
  - Reward mean/std and PPO objective stats per mode.
- Record active batch size per rollout for sanity.
- Ensure `is_teams` is correct in states (fix singles to False).

## Validation & Failure Modes to Guard

- Hidden-state sizing mismatch: initialize using effective active batch size or slice on use.
- Forgetting to ignore controllers for dead ports in singles env → controller send errors.
- Ratio drift: enforce realizability or use per‑env mask; log realized ratio.
- `is_teams` correctness: ensure False for singles; many features/metrics depend on it.
- Reward scale skew: without per‑mode normalization/scales, one mode can dominate gradients.

## Configuration Examples

- 50/50, async, group‑level toggling (Option A)
  - `num_envs=96`, `inner_batch_size=6` → `outer_batch_size=16` (even)
  - `singles_fraction=0.5` → `singles_envs=48` → realizable (48 is multiple of 6)

- 25% singles, async, per‑env mask (Option B)
  - `num_envs=64`, `singles_fraction=0.25` → `singles_envs=16` assigned per env; no divisibility constraint.


