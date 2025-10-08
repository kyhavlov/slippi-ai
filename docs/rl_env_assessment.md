# RL Environment & Training Assessment (2025-10-08)

> **Environment note:** All commands and tests below assume the project’s Linux
> virtual environment at `.linuxvenv`. Activate it with
> `source .linuxvenv/bin/activate` before running anything so dependencies like
> `portpicker`, `fancyflags`, and `ray` resolve correctly on every machine.

## Scope & Entry Points
- `slippi_ai/rl/run_lib.py` builds the RL job: loads policies, wires configs, instantiates `LearnerManager`, and hands off to a `RolloutWorker`.
- `slippi_ai/evaluators.py` (`RolloutWorker`) owns actor construction, delayed-action buffering, and environment orchestration.
- `slippi_ai/envs.py` provides the concrete environment implementations (synchronous, async, legacy Ray wrappers, fake) that ultimately talk to Dolphin.
- `slippi_db/parse_libmelee.py` converts raw `GameState` frames into the 4-player tensor that feeds observations/rewards.
- Reward shaping (`slippi_ai/reward.py`) expects correct `Game` layouts, including a trustworthy `is_teams` flag.

## Current Data Flow Snapshot
1. `run_lib.run` creates Dolphin player configs for ports 1–4 and hands a shared policy state to every port (`run_lib.py:396-424`).
2. `LearnerManager._build_actor` builds a `RolloutWorker` that talks to either `BatchedEnvironment` or `AsyncBatchedEnvironmentMP` depending on `config.actor.async_envs` (`run_lib.py:436-440`).
3. `RolloutWorker` wraps the policy in delayed agents, pushes/pops controller states, and batches per-port trajectories into `Trajectory` tuples (`evaluators.py:68-218`).
4. `LearnerManager._rollout` blindly assumes four logical ports, batching `[1,2,3,4]` every time (`run_lib.py:215-233`).
5. Rewards are recomputed via `reward.compute_rewards`, which looks at `game.is_teams` to normalize singles vs doubles (`reward.py:311-345`).

## Key Findings & Technical Debt

### Environment Orchestration
- `BatchedEnvironment` falls over as soon as `enable_singles=True`. It only provisions `num_envs` UDP ports yet indexes `slippi_ports[i*2 + 1]` (`envs.py:304-327`), leading to out-of-range lookups and unpaired ports.
- `AsyncBatchedEnvironmentMP` attempts a 50/50 split by toggling singles per outer batch (`enable_singles and i % 2 == 0`), but it slices the shared port list in fixed-size chunks (`envs.py:595-608`). Singles envs never receive the second port they expect—this explains the previous "mixed" experiment stalling.
- The top-level `Environment` currently spins up two Dolphin instances when `enable_singles` is set, yet `_current_characters` only tracks the first Dolphin (`envs.py:170-183`), so match reporting loses half the roster in mixed runs.
- `match_reporting.submit_match` assumes a four-name tuple but indexes them inconsistently with `DOUBLES_PORT_MAPPINGS`; swapped ports or singles placeholders mislabel players (`match_reporting.py:38-58`).
- Legacy Ray-based env wrappers at the bottom of `envs.py` are unused and add maintenance burden.
- Excessive `print` debugging (`envs.py:320`, `envs.py:592`, `run_lib.py:425`) pollutes stdout and makes multi-process behavior harder to reason about.

### Observation Layout & Rewards
- Singles frames return with `is_teams=True`, so rewards get scaled as if they were doubles (`parse_libmelee.py:145-154`). That invalidates any singles-vs-doubles balancing knobs in `reward.compute_rewards`.
- Singles placeholders reuse the real opponent object in both `p2` and `p3` depending on `singles_opponent_port` (`parse_libmelee.py:135-140`). There is no explicit "empty" sentinel in the unused opponent slot beyond a dead player template, so downstream code must rely on `is_dead` rather than a structural guarantee.
- `LearnerManager` hard-codes the four-port batch without checking whether an env is singles/doubles, so if we ever change the number of logical trajectories per physical env the learner breaks immediately (`run_lib.py:215-233`).

### Actor Construction & Scheduling
- Name permutations grow as `len(config.agent.name)^4` (`run_lib.py:409-424`). With the default roster this is already >1e6 combos; we only sample `batch_size` of them, but the logic obscures which agent name is controlling which slot and complicates deterministic singles placement.
- The current singles toggle (`config.actor.enable_singles`) exists only on the async code path. Sync envs ignore it entirely, which will surprise anyone trying to run CPU-only smoke tests.

### Instrumentation & Resilience
- `timeout` uses `signal.alarm` inside `Environment.step` (`envs.py:58-86`). In singles mode two dolphins run concurrently, but only the first is guarded by the timeout envelope.
- `Learner.ppo_grads` still prints KL weights on every update (`learner.py:287`), cluttering logs under distributed training.

### Testing Gaps
- No automated test exercises `enable_singles` on `BatchedEnvironment`, `AsyncEnvMP`, or the `RolloutWorker`.
- There is no fixture asserting that singles trajectories come back with exactly one opponent populated and the teammate slot empty.
- Reward tests cover singles replays on the parsing side (`tests/reward_test.py:303-328`) but do not hit the live RL path (Dolphin -> env -> Trajectory -> Learner).

## Stage 0 Preparation Checklist

**Deliverable Breakdown**
1. *Test Scaffolding*: introduce the new harness bits (fake Dolphin/env stubs, deterministic port provider) and land baseline tests that describe today’s behavior.
2. *Port Allocation & Env Wiring*: fix `BatchedEnvironment`/`AsyncEnvMP` singles provisioning and update the environment-contract test expectations accordingly.
3. *Observation Layout Fixes*: correct `Game.is_teams`, ensure opponent placeholders are explicit, and extend the parsing-layout test to cover the new invariants.
4. *Match Reporting & Logging Cleanup*: realign name/character ordering to the mapping, scrub the noisy `print`s, and drop the unused Ray env code.

Each step should be its own commit so problems are easy to bisect, and every code change rides with the corresponding test tweak.

**Targeted Fixes (executed via the breakdown above)**
- Correct `Game.is_teams` for singles frames and ensure unused opponent slots are explicit empties.
- Align `match_reporting` name/character ordering with `DOUBLES_PORT_MAPPINGS` and extend it to handle singles gracefully.
- Replace noisy `print` statements with proper logging (or remove them) and delete the unused Ray-based environment wrappers.

**Test Harness Additions (must land before major refactor)**
1. *Parsing Layout Test*: new unit test covering singles vs doubles `get_game` output (teammate/opponent placement, `is_teams` flag).
2. *Environment Contract Test*: monkeypatch Dolphin with a stub to drive `Environment.current_state` for singles and doubles, verifying we still emit four logical trajectories with consistent placeholders.
3. *Port Wiring Snapshot*: document the current singles limitation by exercising `BatchedEnvironment`/`AsyncBatchedEnvironmentMP` with deterministic fake ports so we can flip the expectation once the Stage 1 scheduler lands.
4. *RolloutWorker Integration Smoke*: run `RolloutWorker` with `use_fake_envs=True`, forcing `enable_singles` on half of the envs and confirming the learner receives balanced singles/doubles batches and `game.is_teams` segmentation.

**Documentation/Tooling**
- Keep this assessment living alongside the RL README so future contributors understand the port/trajectory mapping.
- Add config validation guarding `config.actor.enable_singles` so it only activates when the env implementation supports it.

## Staged Implementation Plan (Post Stage 0)

1. **Stage 1 – Scheduler & Data Plumbing**
   - Move to a true one-Dolphin-per-env model; introduce an explicit scheduler inside `AsyncBatchedEnvironmentMP` (or a new coordinator) that pairs two singles envs into the four-slot batches the learner expects, with deterministic opponent-slot placement.
   - Optionally surface a singles/doubles mode label through `LearnerManager` for observability (not required for mixed training to function).
   - Tests: extend the RolloutWorker smoke test to assert the scheduler’s 50/50 split, controller routing, and ordering.

2. **Stage 2 – Learner & PPO Adjustments**
   - Update advantage/value calculations to operate on mixed-mode batches (ensure reward normalization matches the new `is_teams` flag).
   - Surface telemetry in wandb for per-mode reward, KL, and PPO objectives so regressions are visible.
   - Tests: add learner-level regression coverage feeding synthetic singles+doubles trajectories through `Learner.ppo` and validating gradient magnitudes/metrics.

3. **Stage 3 – Runtime & Ops**
   - Wire CLI flags to configure singles:doubles ratios (default 50/50) and document expectations in the RL scripts.
   - Exercise full pipeline smoke (fake envs + short PPO loop) under the new config, then provide guidance for real Dolphin ops.

## Immediate Next Steps
- Execute the Stage 0 breakdown above. First deliverable is the test scaffolding commit; once the new tests exist and describe current behavior we can review them together before progressing to the fixes.
- After Stage 0 tests are green, reconvene to lock the Stage 1 scheduler design and unblock implementation.
