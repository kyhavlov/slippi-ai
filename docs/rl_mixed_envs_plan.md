# Mixed Singles + Doubles RL: Design and Implementation Plan

Author: LLM planning pass on 2025-09-26

Status: Planning only (no code changes yet)

Scope: Enable robust RL training with a controllable mix of singles and doubles environments, targeting a default 50/50 trajectory split. Remove fragile multi-Dolphin-per-env behavior and make shapes, rewards, and logging explicitly support mixed-mode training.

---

## Goals

- Train with a configurable fraction of singles vs doubles during RL (default 50/50) to improve robustness and skill balance.
- Use exactly one Dolphin process per environment for both modes to simplify resets/lifecycle and isolate failures.
- Keep the existing 4-slot `Game` schema for all environments to minimize invasive changes in the actor/evaluator pipeline.
- Prevent dummy ports from polluting gradients by masking inactive columns in the learner.
- Make reward scales comparable across modes (configurable), and add per‑mode logging to monitor stability and progress.

## Decisions

1) One-Dolphin-per-env for both singles and doubles
   - Singles envs expose a 4-slot `Game` with real players in `p0` (self) and `p2` (opponent); `p1` and `p3` are dead placeholders.
   - `is_teams=False` for singles env states; `is_teams=True` for doubles.
   - Env ignores controllers for `p1`/`p3` in singles mode.

2) Keep 4 agents/ports at the rollout layer; add an active-column mask
   - Don’t reshape the pipeline to 2 ports for singles; instead, provide a per‑column `active_mask` so the learner trains only on live ports.
   - Effective batch columns per rollout become `4 * (#doubles_envs) + 2 * (#singles_envs)`.

3) Ratio control via `singles_fraction`
   - Replace `enable_singles` (bool) with `singles_fraction: float` on the actor config.
   - Validate realizability given `num_envs` and batching strategy (see below). Default to 0.5.

4) Reward normalization by mode
   - Support separate `RewardConfig` for singles vs doubles.
   - Add optional per‑mode scaling (simple multiplier) and/or per‑mode advantage normalization (preferred).

5) Observability
   - Fix `is_teams` values for RL envs and log per‑mode metrics: realized mix ratio, FPS/MPS, actor_kl, teacher_kl, UEV, reward mean/std, PPO objective.

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
    - `p2`: opponent
    - `p3`: placeholder dead player
    - `is_teams=False`
  - `step` ignores incoming controllers for `p1` and `p3`.

- Doubles env:
  - One Dolphin instance as today.
  - `is_teams=True`.

### Ratio control & provisioning

- `singles_fraction: float` (0..1) on `ActorConfig` replaces `enable_singles`.
- Compute `singles_envs = round(num_envs * singles_fraction)`.
- Provision exactly one port per env (`num_envs` total), no extra provisioning for singles.
- Assignment policy:
  - Option A (simple): keep group-level toggling for async envs; require `singles_envs` be a multiple of `inner_batch_size`. Validate at startup; fail fast otherwise.
  - Option B (better): per‑env toggling with a boolean mask passed to the env builder; removes divisibility constraints, but requires minor plumbing to deliver a per‑env mode flag.

### Rollout packaging & learner masks

- Continue to build 4-port trajectories for every env as today.
- Add `active_mask: np.ndarray[bool]` to `evaluators.Trajectory` with shape `[B]` (one boolean per batch column):
  - For doubles columns (p0,p1,p2,p3) → True.
  - For singles columns → True for p0,p2; False for p1,p3.
- Learner changes:
  - When computing PPO, first gather/slice all frame tensors, actions, advantages, and is_resetting using `active_mask` along the batch dimension.
  - Hidden-state sizing: compute the effective batch size at init as the sum of active columns from the dummy trajectory for the configured ratio (or infer per rollout and handle via masked slicing).
  - Maintain variable shapes otherwise; this is a selection (mask) operation, not a reshape.

### Rewards and normalization

- Add config:
  - `learner.reward_singles: RewardConfig` (defaults to current values)
  - `learner.reward_doubles: RewardConfig` (defaults to current values)
  - `learner.reward_scale_singles: float = 1.0`
  - `learner.reward_scale_doubles: float = 1.0`
  - `learner.normalize_advantages_per_mode: bool = True`

- Computation path:
  1) Compute base rewards via `compute_rewards` as today.
  2) Split batch columns by mode using `states.is_teams[0]`.
  3) Apply per‑mode `RewardConfig` and optional scale.
  4) Compute returns/advantages; if enabled, z‑normalize advantages per mode (zero mean, unit variance with epsilon).

Note: A simple global multiplier (e.g., “divide by 2 in doubles while teammate alive”) is acceptable as `reward_scale_doubles=0.5`, but per‑mode configs + per‑mode advantage normalization is preferred for stability and clarity.

### Logging & metrics

- Record per‑mode aggregates each flush (in `rl/run_lib.py`):
  - Realized singles share, FPS/MPS, actor_kl mean/max, teacher_kl, UEV.
  - Reward mean/std and PPO objective stats per mode.
- Record active batch size per rollout for sanity.
- Ensure `is_teams` is correct in states (fix singles to False).

## Implementation Phases (minimal risky steps first)

Phase 0 — Hardening + Visibility
- Add `singles_fraction` to `ActorConfig`. Keep `enable_singles` as a deprecated alias that maps `False→0.0`, `True→0.5`.
- Validate `singles_fraction` realizability for Option A (group-level) or implement Option B mask.
- Fix `is_teams` for singles states in RL env path.
- Add per‑mode logging and realized ratio reporting (no behavior change yet).

Phase 1 — One-Dolphin Singles
- Remove dual-Dolphin creation in `Environment` for singles; always one Dolphin per env.
- In singles env, ignore controllers for `p1/p3` and synthesize dead placeholders in `current_state`.
- Port provisioning: exactly `num_envs` UDP ports.

Phase 2 — Active Mask & Learner Integration
- Add `active_mask` to `evaluators.Trajectory` and plumb it from env metadata.
- In the learner (`rl/learner.py`):
  - Before PPO, mask batch columns in frames/actions/advantages by `active_mask`.
  - Recompute effective batch size for init/hidden-state handling (or use masking-friendly init with maximum size and slice on use).

Phase 3 — Reward Normalization
- Add `reward_singles`, `reward_doubles`, `reward_scale_*`, and `normalize_advantages_per_mode` to the learner config.
- Branch rewards/advantages per mode using `states.is_teams` or the env mask.

Phase 4 — Tests
- Provisioning tests (async path): for tuples `(num_envs, inner_batch_size, singles_fraction)` assert:
  - Singles count matches request.
  - Port count equals `num_envs`.
  - First popped state shows correct `is_teams` mask.
- Rollout tests:
  - Short rollouts yield correct `active_mask` counts (2 per singles env, 4 per doubles env).
  - Learner can complete PPO steps with mixed trajectories.
  - Per‑mode logs present with non‑NaN values.

## Validation & Failure Modes to Guard

- Hidden-state sizing mismatch: initialize using effective active batch size or slice on use.
- Forgetting to ignore controllers for dead ports in singles env → controller send errors.
- Ratio drift: enforce realizability or use per‑env mask; log realized ratio.
- Sync envs: either explicitly unsupported for mixed mode or fully provisioned (one Dolphin per env always → simpler to support sync as well).
- `is_teams` correctness: ensure False for singles; many features/metrics depend on it.
- Reward scale skew: without per‑mode normalization/scales, one mode can dominate gradients.

## Configuration Examples

- 50/50, async, group‑level toggling (Option A)
  - `num_envs=96`, `inner_batch_size=6` → `outer_batch_size=16` (even)
  - `singles_fraction=0.5` → `singles_envs=48` → realizable (48 is multiple of 6)

- 25% singles, async, per‑env mask (Option B)
  - `num_envs=64`, `singles_fraction=0.25` → `singles_envs=16` assigned per env; no divisibility constraint.

## Rollout & Performance Notes

- With one Dolphin per env, singles no longer require extra processes; resource usage becomes linear in `num_envs` for any ratio.
- Two inactive columns per singles env still flow through the actor; masking prevents their contribution to PPO, but we still pay some inference cost. Future optimization could skip running those agents entirely for singles envs.

## File Touch Points (for implementation later)

- Env layer: `slippi_ai/envs.py`
  - Environment: single Dolphin per env (singles/doubles); ignore p1/p3 controllers in singles; set `is_teams` correctly.
  - Batched/Async env builders: pass per‑env singles mask or enforce group-level ratio; allocate exactly `num_envs` ports.

- Evaluator: `slippi_ai/evaluators.py`
  - Add `active_mask` to `Trajectory` and construct it when batching trajectories across ports/envs.

- RL runner: `slippi_ai/rl/run_lib.py`
  - Add `ActorConfig.singles_fraction` and config validation.
  - Extend logging to emit per‑mode metrics and realized ratio.

- Learner: `slippi_ai/rl/learner.py`
  - Accept `active_mask` and mask batch columns for PPO.
  - Add reward config by mode and per‑mode advantage normalization.

- Reward: `slippi_ai/reward.py`
  - No core changes required; scaling is applied in learner prior to advantage computation. Optional: mode‑specific configs can be selected here if preferred.

- Scripts: `scripts/rl_doubles.sh`
  - New flag `--config.actor.singles_fraction=0.5` (keep old `enable_singles` as deprecated alias).

## Open Questions

- Do we want synchronous env support for mixed mode or make mixed mode async‑only for now? (Recommendation: async‑only initially.)
- Should we skip building agents for inactive ports in singles to save inference compute? (Would require actor refactor; defer.)
- Do we want to expose an automatic ratio annealing schedule (e.g., start with more doubles, then mix in singles)?

## Next Steps (when we resume)

1) Implement Phase 0 (config + logging + `is_teams` fix) and run a short smoke to collect per‑mode metrics with the current code.
2) Implement Phase 1 (one Dolphin per env) and validate provisioning/ports; re‑run smoke tests.
3) Implement Phase 2 (active_mask + learner masking); add minimal tests; confirm PPO runs with mixed data.
4) Implement Phase 3 (per‑mode reward config/normalization) and verify gradient scales in logs.
5) Land test suite additions and document launch guidance in `AGENTS.md`/scripts.

