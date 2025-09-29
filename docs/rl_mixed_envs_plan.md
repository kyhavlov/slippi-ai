# Mixed Singles + Doubles RL: Design and Implementation Plan

Author: LLM planning pass on 2025-09-29

Status: Planning only (no code changes yet)

Scope: Enable robust RL training with a controllable mix of singles and doubles environments, targeting a default 50/50 trajectory split. Remove fragile multi-Dolphin-per-env behavior and make shapes, rewards, and logging explicitly support mixed-mode training.
Much of the core of these changes will likely live in `slippi_ai/rl/run_lib.py` and `slippi_ai/envs.py`.

---

## Goals

- Train with a controllable mix of singles vs doubles during RL (defaulting to a 50/50 split) to improve robustness and skill balance.
- Use exactly one Dolphin process per environment for both modes to simplify resets/lifecycle and isolate failures.
- Keep the existing 4-slot `Game` schema for all environments to minimize invasive changes in the actor/evaluator pipeline.
- Prevent dummy ports from polluting gradients by masking inactive columns in the learner.
- Make reward scales comparable across modes (configurable), and add per‑mode logging to monitor stability and progress.

## Decisions

1) One-Dolphin-per-env for both singles and doubles
   - Singles envs expose a 4-slot `Game` with real players in `p0` (self) and the opponent in an even split of either `p2` or `p3` (to avoid bias when carrying over knowledge to doubles); `p1` and the unused opponent slot are dead placeholders.
   - `is_teams=False` for singles env states; `is_teams=True` for doubles.
   - Env ignores controllers for `p1` and the unused opponent slot in singles mode.

2) Keep 4 agents/ports at the rollout layer; add an active-column mask
   - Don’t reshape the pipeline to 2 ports for singles; instead, provide a per‑column `active_mask` so the learner trains only on live ports. Need to carefully test this both locally and end-to-end to make sure the exact correct trajectories and their shapes make it to the learner in the state we expect.
   - Effective batch columns per rollout become `4 * (#doubles_envs) + 2 * (#singles_envs)`.
   - Need to look into disabling inference for inactive singles ports, will likely require some refactoring in the actor/agent code but is worth it as inference time is our bottleneck during rollout generation currently.

3) Ratio control via `singles_fraction`
   - Replace `enable_singles` (bool) with `singles_fraction: float` on the actor config.
   - Default to 0.5 (even split). Implementation will handle arbitrary ratios via a per-env mask, but initial configs can assume 50/50 for simplicity.

4) Reward normalization by mode
   - Rely on existing `team_size_normalization` for now. Add per-mode reward logging so we can revisit scaling if magnitude drift shows up.

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
- Default computation: `singles_envs = num_envs // 2` (even split). The builder produces a deterministic boolean mask whose first `singles_envs` entries are singles; later we can expand to arbitrary ratios by changing only the mask generator.
- Provision exactly one dolphin port per env (`num_envs` total), no extra provisioning for singles.
- Pass the per-env singles mask into `SafeEnvironment`/`BatchedEnvironment`/async/ray variants so each env instance knows its mode.
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

- Use `team_size_normalization` (already in `RewardConfig`) to keep per-frame magnitudes comparable.
- Add logging to surface singles vs doubles reward means/stds; revisit explicit scaling only if the data warrants it.

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

## Configuration Example

- 50/50 split (default): `num_envs=96`, `singles_envs=48`, mask `[True]*48 + [False]*48`; works for sync and async actors because each env reads its own mode flag.

---

## Updated Implementation Plan (2025-09-29)

This supersedes the earlier outline. Steps will be executed sequentially, each with targeted validation before progressing.

1. **Config plumbing for singles mix** (`slippi_ai/rl/run_lib.py`, `slippi_ai/rl/config_utils.py`, `slippi_ai/rl/run.py`, launch scripts)
   - Replace `enable_singles` with `singles_fraction` (float). Default to 0.5 (reset to 0.0 in `DEFAULT_CONFIG`).
   - Add a helper that, given `(num_envs, singles_fraction)`, returns a deterministic single/double mask (current implementation rounds to the nearest env count and marks the leading indices as singles; more flexible placement can come later).
   - Thread the mask through config serialization/deserialization and CLI flags. Update helper scripts to pass the new flag. Until Step 2 lands, async actors still flip the legacy `enable_singles` flag internally but use the new mask for validation.
   - Tests: unit test for the mask helper to verify counts/order and error handling; adjust any config round-trip tests.

2. **Environment lifecycle refactor** (`slippi_ai/envs.py` and builders)
   - Refactor `SafeEnvironment`/`Environment` to run exactly one Dolphin per env regardless of mode; remove `slippi_port2` plumbing.
   - Accept a per-env boolean (`is_singles`) and derive the correct controller/port wiring. For singles, ensure placeholder players are marked dead and ignored, and set `is_teams=False`.
   - Update batched/async/ray builders to iterate over the mask when instantiating environments.
   - Tests: lightweight fake-dolphin test ensuring singles envs create one Dolphin and produce 4-slot games with dead placeholders and `is_teams=False`.

3. **Activity metadata propagation** (`slippi_ai/envs.py`, `slippi_ai/evaluators.py`)
   - Compute an `active_ports` boolean array per env (ports that represent real players) alongside the `Game` data.
   - Extend `evaluators.Trajectory` with an `active_mask` (batch-major) and ensure `Trajectory.batch`/`dummy_trajectory` include it.
   - Tests: unit test batching behavior for the mask (mix singles/doubles trajectories and check concatenation).

4. **Actor/agent batching overhaul** (`slippi_ai/evaluators.py`, `slippi_ai/rl/run_lib.py`)
   - For each port, gather indices of envs where that port is active and instantiate `DelayedAgent` with that smaller batch size.
   - Before calling `agent.push`, slice env states/needs_reset to the active indices; after inference, scatter controllers back, filling inactive slots with neutral inputs.
   - Keep nametag batching aligned with the same index lists.
   - Tests: fake-agent test confirming inactive ports never trigger inference and controller scatter preserves zeros.

5. **Learner pipeline masking** (`slippi_ai/rl/run_lib.py`, `slippi_ai/rl/learner.py`)
   - Derive the effective batch size from the mask and initialize learner hidden states accordingly.
   - Introduce utilities to slice trajectory tensors by the mask before computing PPO/value updates.
   - Ensure checkpoints capture any additional mask metadata if necessary.
   - Tests: synthetic-trajectory learner test comparing masked vs manually pre-sliced runs.

6. **Metrics and logging updates** (`slippi_ai/rl/run_lib.py`)
   - Log realized singles ratio, per-mode reward mean/std, per-mode PPO metrics (actor_kl, teacher_kl, UEV), and effective active batch size during flushes.
   - Validate `is_teams` stats now differentiate singles/doubles correctly.
   - Tests: extend logger tests (or add new) validating the metrics dictionary contains the expected per-mode keys.

7. **Docs and script refresh** (this document, `scripts/rl_*.sh`, README snippets)
   - Document new flags and assumptions (50/50 default, team-size normalization reliance).
   - Update example scripts to use `singles_fraction` and remove legacy `enable_singles` references.
   - Tests: `python -m compileall` on updated scripts to catch syntax issues.

8. **End-to-end validation**
   - Provide a fake-env smoke command (e.g., `python slippi_ai/rl/run.py --config.actor.use_fake_envs=True --config.actor.num_envs=4 --config.actor.singles_fraction=0.5`) to verify actors build, masks propagate, and PPO runs at least one epoch.
   - Document an optional short real-env sanity run for when Dolphin access is available.

Each step ends with targeted verification before moving on, and we’ll revisit reward scaling only if the logged per-mode stats expose a mismatch after implementation.
