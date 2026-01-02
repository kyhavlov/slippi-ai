#!/usr/bin/env bash
set -euo pipefail

ROOTDIR="$(pwd)"

if [ -f "$ROOTDIR/.linuxvenv/bin/activate" ]; then
  # Project-local venv (see AGENTS.md).
  # shellcheck disable=SC1091
  source "$ROOTDIR/.linuxvenv/bin/activate"
fi

export PYTHONPATH="."

TAG="fake_rl_perf_$(date +%Y%m%d_%H%M%S)"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-90}"
TEACHER_PATH="${TEACHER_PATH:-$ROOTDIR/models/imitation_v17_2000k.pkl}"
EXPT_ROOT="${EXPT_ROOT:-/tmp/slippi_ai_bench}"

LOG_PATH="${LOG_PATH:-/tmp/${TAG}.log}"

echo "tag=$TAG log=$LOG_PATH timeout_s=$BENCH_TIMEOUT_S teacher=$TEACHER_PATH" >&2

timeout --signal=TERM "${BENCH_TIMEOUT_S}"s python -u slippi_ai/rl/run.py \
  --config.runtime.tag="$TAG" \
  --config.runtime.expt_root="$EXPT_ROOT" \
  --config.runtime.max_step=10 \
  --config.runtime.log_interval=5 \
  --config.runtime.save_interval=1000000 \
  --config.teacher="$TEACHER_PATH" \
  --config.opponent.type=self \
  --config.opponent.train=True \
  --config.actor.use_fake_envs=True \
  --config.actor.num_envs=48 \
  --config.actor.rollout_length=240 \
  --config.actor.gpu_inference=True \
  --config.actor.fuse_ports_inference=True \
  --config.agent.name_allowlist="Fox:Master Player" \
  --config.optimizer_burnin_steps=0 \
  --config.value_burnin_steps=0 \
  --wandb.mode=disabled \
  "$@" \
  2>&1 | tee "$LOG_PATH"

# Print the most recent timings line if present.
rg -n "fps:|timings" "$LOG_PATH" | tail -n 5 || true

