# Slippi-AI Agents Guide

## Mission & Scope
- This is a SSBM ai project started by vladfi1 and forked by me with modifications to support 2v2.
- Goal: keep pipeline maintainable for human + LLM contributors spanning data prep, imitation, RL, evaluation, and analytics.
- Default branch currently targets the doubles extension; legacy singles code remains but may diverge.

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
- Python 3.8 environment (`environment.yaml:1`), TensorFlow 2.6 + Sonnet, ray, fancyflags/absl, wandb.
- GPU access assumed for training; RL scripts tuned for RTX 3080Ti but configurable via CLI.
- External binaries kept in repo (Slippi AppImages, `SSBM.iso`) – do not redistribute; avoid accidental commits.
- Dockerfile (`Dockerfile:1`) offers minimal ray + TF baseline; still requires manual peppi-py alignment for parsing.
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
2. Monitor wandb (`eval.policy.loss` plateau indicates convergence).
3. Outputs land in `experiments/<tag>/latest.pkl`; optionally sync to S3 via `Config.save_to_s3`.

### 3. Reinforcement Learning
1. Start from imitation checkpoint (`--config.teacher` in `scripts/rl_doubles.sh:28`).
2. Tune env counts / rollout length according to hardware; optional singles mix via `--config.actor.enable_singles`.
3. Self-play stats stream to wandb; checkpoints rotate under `experiments/rl/<tag>`.

### 4. Evaluation & Deployment
- `scripts/eval_doubles.py:1` pits agents vs each other/humans using `DOLPHIN_PATH`/`ISO_PATH`.
- `scripts/netplay_doubles.py:1` drives live online matches; needs `teams_connect_code` and Slippi user JSON.
- Match results post to dashboard via `slippi_ai/match_reporting.py:38`; ensure Flask app running (`dashboard/dashboard.py:19`).

### 5. Analytics & Dashboard
- Launch Flask app (`python dashboard/dashboard.py`) to visualize JSONL stats; data auto-appends per match submission.
- `dashboard/static/` + `templates/` hold frontend; keep CSS/JS minimal for LLM diffs.

## Testing & Validation
- Python tests under `tests/` (e.g., `tests/networks_test.py:1`, `tests/rl_lib_test.py:1`); run via `pytest` or targeted scripts.
- Shell harnesses (`tests/train_two.sh:1`, `tests/training_test.sh:1`) sanity-check CLI entrypoints.
- `test_output.sh` and `test_imports.py` help verify environment imports without full training.
- Prefer GPU-offline smoke runs before long jobs; set `CUDA_VISIBLE_DEVICES=""` for CPU-only sanity tests.

## Contribution Practices for LLMs
1. **Start with reconnaissance** – use `find`, `grep`, `python -m compileall` to confirm context before editing.
2. **Favor minimal, well-scoped diffs** – touch smallest module slice; update comments/docstrings sparingly.
3. **Respect configs & flags** – new options should integrate with fancyflags/absl conventions; update example scripts.
4. **Document large-impact changes** – extend this guide, adjust README, annotate scripts when altering workflows.
5. **Validate** – run unit/smoke tests relevant to touched modules; note skipped tests and why.
6. **Preserve data hygiene** – never commit regenerated parquet/checkpoints; add `.gitignore` rules when needed.

## Observed Opportunities / TODO Seeds
- Consolidate singles/doubles env handling (feature flags in `slippi_ai/envs.py:80`).
- Expand automated tests for doubles RL (currently sparse beyond smoke tests).
- Normalize script duplication (`scripts/online_doubles*.sh`) into param-driven templates.
- Document dashboard API contract and hard-coded URL in `slippi_ai/match_reporting.py:6`.
- Evaluate migrating from TensorFlow to JAX/PyTorch if long-term maintenance demands (requires major refactor).

## Quick Reference
- Launch imitation: `./scripts/imitation_doubles.sh --config.dataset.meta_path=data/meta.json`
- Launch RL: `./scripts/rl_doubles.sh --config.teacher=/path/to/latest.pkl`
- Local eval: `python scripts/eval_doubles.py --p1.ai.path=... --p2.ai.path=...`
- Netplay: `python scripts/netplay_doubles.py --dolphin.user_json_path=bot1-user.json --dolphin.teams_connect_code=ABCD#123`
- Dashboard: `python dashboard/dashboard.py` (serves on `http://127.0.0.1:5000` by default)

