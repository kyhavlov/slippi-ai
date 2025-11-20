# Slippi-AI Agents Guide

> **Environment setup reminder:** on Linux machines use the project virtual
> environment at `.linuxvenv` (`source .linuxvenv/bin/activate`) before running
> scripts or tests so TensorFlow, portpicker, fancyflags, etc. resolve
> correctly.

## Rules
- Do not ever add 'skip' annotations to tests, for any reason (dependencies missing or otherwise). If you can't figure out a dependency error, ask me to help resolve it.
- Do not ever add try/except around imports. Imports MUST succeed, period. If they don't we will fix that root problem. Adding conditionals around import success is UNACCEPTABLE.

## Mission & Scope
- This is a SSBM ai project started by vladfi1 and forked by me with modifications to support 2v2.
- Goal: keep pipeline maintainable for human + LLM contributors spanning data prep, imitation, RL, evaluation, and analytics.
- Currently targets the doubles extension; legacy singles code remains but may diverge.

## Systems Overview
- **Data ingestion** – parse Slippi replays into parquet + metadata (`slippi_db/parse_local.py:1`) feeding imitation datasets in `data/`.
- **Imitation learning** – supervised training loop driven by `slippi_ai/train_lib.py:2` and launched via `scripts/imitation_doubles.sh:1` / `scripts/train.py`.
- **Reinforcement learning** – PPO-style self-play pipeline (`slippi_ai/rl/run_lib.py:1`, `slippi_ai/rl/train_two.py`) launched with `scripts/rl_doubles.sh:1`.
- **Evaluation & deployment** – local eval (`scripts/eval_doubles.py:1`), online netplay (`scripts/netplay_doubles.py:1`), Discord/Twitch hooks under `scripts/`.
- **Analytics & reporting** – match summaries emitted through `slippi_ai/match_reporting.py:1` into the Flask dashboard (`dashboard/dashboard.py:1`).

## Repository Map (high-value dirs)
- `slippi_ai/` – Core models, policies, environments, saving, match reporting, utilities.
- `slippi_ai/rl/` – RL configs, learner loop, actor wiring; self-play logic centralized here.
- `slippi_db/` – Replay parsing, preprocessing, Ray cluster configs, S3 helpers for datasets.
- `scripts/` – Launchers for imitation, RL, evaluation, profiling, live services, sync helpers.
- `dashboard/` – Flask app + static assets for match analytics; writes JSONL to `dashboard/melee_data/`.
- `data/` – Local dataset cache (Raw/Parsed/meta). Treat as large, mostly-generated artifacts.
- `tests/` – Smoke/unit tests for core libs plus shell harnesses (`tests/unit_tests.py:1`, `tests/train_two.sh:1`).
- `discordbot/`, `bot*-user.json` – Netplay and streaming integration assets; keep credentials out of version control.

## Tooling & Dependencies
- Python environment, TensorFlow + Sonnet, ray, fancyflags/absl, wandb.
- GPU access assumed for training; RL scripts tuned for RTX 3080Ti but configurable via CLI.
- External binaries kept in repo (Slippi AppImages, `SSBM.iso`) – do not redistribute; avoid accidental commits.
- Set `PYTHONPATH=.` when running scripts; wandb requires `WANDB_API_KEY`.

## Data & Asset Management
- Replay pipeline expects `data/Raw`, `data/Parsed`, `data/meta.json` per `slippi_db/parse_local.py:1`.
- Singles replays can be folded into doubles training (`slippi_ai/data.py:159`); ensure metadata marks `is_singles`.
- Large checkpoints live under `experiments/` and `models/`; prune before committing.
- Sync scripts (`sync_to_vast.sh`, `sync_to_d2.sh`) push/pull to remote storage—verify credentials and dry-run.

## Core Workflows
### 1. Build / Update Dataset
1. Drop .slp archives into `data/Raw`.
2. Run `python slippi_db/parse_local.py --root=data` for parquet generation.
3. Refresh metadata (`scripts/make_local_dataset.py` if present) to regenerate `meta.json`.

### 2. Imitation Training
1. Configure dataset + filters via CLI (`scripts/imitation_doubles.sh:1`) or `scripts/train.py`.
   - Set `BALANCE_CHARACTERS=1` (or pass `--config.data.balance_characters=True`) when you need replay sampling to be even across characters. Leaving it unset preserves raw dataset frequency.
2. Monitor wandb (`eval.policy.loss` plateau indicates convergence).
3. Outputs land in `experiments/<tag>/latest.pkl`; optionally sync to S3 via `Config.save_to_s3`.

### 3. Reinforcement Learning
1. Start from imitation checkpoint (`--config.teacher` in `scripts/rl_doubles.sh:28`).
2. Tune env counts / rollout length according to hardware. Mixed singles mode is currently removed; RL runs doubles-only until a new implementation lands.
3. Self-play stats stream to wandb; checkpoints rotate under `experiments/rl/<tag>`.

### 4. Evaluation & Deployment
- `scripts/eval_doubles.py:1` pits agents vs each other/humans using `DOLPHIN_PATH`/`ISO_PATH`.
- `scripts/netplay_doubles.py:1` drives live online matches; needs `teams_connect_code` and Slippi user JSON.
- Match results post to dashboard via `slippi_ai/match_reporting.py:38`; ensure Flask app running (`dashboard/dashboard.py:19`).

### 5. Analytics & Dashboard
- Launch Flask app (`python dashboard/dashboard.py`) to visualize JSONL stats; data auto-appends per match submission.
- `dashboard/static/` + `templates/` hold frontend; keep CSS/JS minimal for LLM diffs.
- `python scripts/value_trace.py --model_path=... --replay_path=...` plots value-function predictions vs. realized rewards for a replay (outputs PNG + CSV).

## Testing & Validation
- Python tests under `tests/` (e.g., `tests/networks_test.py:1`, `tests/rl_lib_test.py:1`); run via `pytest` or targeted scripts.
- Shell harnesses (`tests/train_two.sh:1`, `tests/training_test.sh:1`) sanity-check CLI entrypoints.
- `test_output.sh` and `test_imports.py` help verify environment imports without full training.
- Prefer GPU-offline smoke runs before long jobs; set `CUDA_VISIBLE_DEVICES=""` for CPU-only sanity tests.
- Test harnesses use `python -m unittest`, not `pytest`; keep that in mind when adding new tests or giving run instructions.
- Dev installs currently rely on Vlad’s `peppi`/`peppi-py` `dev` branches (Rust nightly, edition2024). On a fresh env:
  1. `rustup toolchain install nightly` (once) and either `export RUSTUP_TOOLCHAIN=nightly` for your shell or `rustup override set nightly` in a temp build dir.
  2. Clone both repos:
     * `git clone https://github.com/vladfi1/peppi.git` && `git checkout dev`
     * `git clone https://github.com/vladfi1/peppi-py.git` && `git checkout dev`
  3. From the venv: `pip install --upgrade /path/to/peppi-py` (the build uses the sibling `peppi` repo).
  4. Verify with `python -m pip show peppi-py` (should report 0.8.2).

## Contribution Practices for LLMs
1. **Start with reconnaissance** – use `find`, `rg`, `python -m compileall` to confirm context before editing.
2. **Prefer ripgrep** – use `rg` for repo-wide searches; avoid raw `grep -R` so traversals stay fast.
3. **Favor minimal, well-scoped diffs** – touch smallest module slice; update comments/docstrings sparingly.
4. **Respect configs & flags** – new options should integrate with fancyflags/absl conventions; update example scripts.
5. **Document large-impact changes** – extend this guide, adjust README, annotate scripts when altering workflows.
6. **Validate** – run unit/smoke tests relevant to touched modules; note skipped tests and why.
7. **Preserve data hygiene** – never commit regenerated parquet/checkpoints; add `.gitignore` rules when needed.

## Observed Opportunities / TODO Seeds
- Consolidate singles/doubles env handling (feature flags in `slippi_ai/envs.py:80`).
- Expand automated tests for doubles RL (currently sparse beyond smoke tests).
- Normalize script duplication (`scripts/online_doubles*.sh`) into param-driven templates.
- Document dashboard API contract and hard-coded URL in `slippi_ai/match_reporting.py:6`.
- Evaluate migrating from TensorFlow to JAX/PyTorch if long-term maintenance demands (requires major refactor).

## Upstream Imitation-Dev Audit (2025-09-25)
- Items, FoD platforms, and Randall data now flow end-to-end (types, parsers, embeddings) with optional MLP processing for item slots; see upstream commits `5bf25ba`, `ecd9811`, `c728ffe`, `339380e`, `98466c5`, `fef74f8`.
- Ice Climbers coverage includes Nana parsing/embedding and jump one-hot fixes plus config upgrades (`c6b2277`, `855d535`, `d750a33`, `ae587b2`).
- Data pipeline adds character-balanced sampling, cached reward computation, dataset metrics, and per-character eval logging (`02e6f09`, `a88636a`, `2f10a2f`, `d988fa2`, `27c8fcf`).
- Replay parsing tightens rollback handling, processed buttons, FoD platform exposure, and upgrade tooling (`f5208b0`, `d0e1607`, `862647c`, `596f847`, `ee2d7dc`).
- New observation and config plumbing (e.g., `slippi_ai/observations.py`, `saving.py` v5 upgrades) would need reconciliation with doubles-specific structures before merging.

## Roadmap (done)
- Pull in some of the upstream improvements from https://github.com/vladfi1/slippi-ai on the imitation-dev branch: item/projectile embeddings, nana embedding per-player, randall embedding (have one currently but it's probably incorrectly done/suboptimal), support for balancing replay data per-character during imitation learning. Probably other small improvements as well in the history of that branch. When adding new embeddings or changing existing ones, follow the existing conventions of adding them as optional, configurable fields to maintain backwards compatibility with running older versions of models. See [Upstream Imitation-Dev Audit (2025-09-25)](#upstream-imitation-dev-audit-2025-09-25) for commit-level notes before starting integration work.
- Minor upstream improvement: pull changes to enable kirby and update our local libmelee fork to get upstream changes there for it too.
- Double check that the player embeddings are properly voided for the singles games we use for imitation training. This would be in both the replay pre-processing and the IL code to double check the gamestate the model gets makes it obvious in some way that the player is not there/eliminated. I added an 'is_teams' embedding at some point but i'm not sure it's well done.
- Add support for splitting training between some % singles and some % doubles games during RL. Last time i looked into this it was tricky because of how the batches from the envs are lined up/prepared, and having envs of different sizes (2 vs 4 players each) seemed to present complications. This should be doable though, but it will take some care and testing.
- Set up a script to load a trained model and run a replay through it in order to log its value function from one player's perspective throughout the game. Emit a readable graph as well. (done, in scripts/value_trace.py)

## Roadmap (TODO)
- Experiment with smaller network size for faster inference/training.
- Add a moderate penalty (0.002 per frame or so) for existing as Zelda instead of Sheik, to hard incentivize transforming off Zelda when able.
- Look at adding a win probability head to the model (possibly with shared trunk or separate, idk which is better). Ideally we would incorporate this in RL in some way to surface the true game win signal/reward to the model so it's able to think long term better for things like beneficial trades or stock 1-for-1s. Not sure how exactly this should work.
- Equalize all character-env balance during RL, dont need separate per-character distribution
- Comprehensive review of reward function.
  - Full audit of how it works, pitfalls, bugs, inconsistencies, potential improvements.
  - Begin by adding a bunch of tests to verify current behavior
  - look into optimizing its performance (some work done in upstream for this i think, should review that and potentially apply gains)
  - add a 'Zelda penalty' for existing as zelda per-frame to incentivize it to switch back to sheik quickly. 
  - Normalize reward between singles/doubles games somehow, not sure exactly how this should work.
- Once reward audit/cleanup is done, can start thinking about adding proper support for mixed singles+doubles envs during RL. Partially implemented but not done well, and not working. Some planning outlined in docs/rl_mixed_envs_plan.md

## Quick Reference
- Launch imitation: `./scripts/imitation_doubles.sh --config.dataset.meta_path=data/meta.json`
- Launch RL: `./scripts/rl_doubles.sh --config.teacher=/path/to/latest.pkl`
- Local eval: `python scripts/eval_doubles.py --p1.ai.path=... --p2.ai.path=...`
- Netplay: `python scripts/netplay_doubles.py --dolphin.user_json_path=bot1-user.json --dolphin.teams_connect_code=ABCD#123`
- Dashboard: `python dashboard/dashboard.py` (serves on `http://127.0.0.1:5000` by default)
