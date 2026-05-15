#!/usr/bin/env sh
set -eu

OUTDIR="reports/triage/jax_sim_rl_runs/imitation_v19_sim_foxditto_lr3e5_kl5e2_pb1_burn8_100"
mkdir -p "$OUTDIR"

exec env -u LD_LIBRARY_PATH \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  PYTHONPATH=. \
  .venv/bin/python scripts/benchmark_jax_sim_rl.py \
    --model-path models/imitation_v19.pkl \
    --workers 16 \
    --batch-size 512 \
    --matchup fox-fox \
    --rollout-length 80 \
    --ppo-batches 1 \
    --ppo-epochs 1 \
    --learner-minibatch-size 1024 \
    --learner-minibatch-scan-size 32 \
    --learning-rate 3e-5 \
    --policy-gradient-weight 3 \
    --kl-teacher-weight 5e-2 \
    --value-cost 1 \
    --reward-halflife 8 \
    --reward-stalling-penalty 0.1 \
    --reward-stalling-threshold 50 \
    --reward-approaching-factor 1e-3 \
    --reward-ledge-grab-penalty 0.02 \
    --reward-zelda-penalty 0.01 \
    --ppo-beta 0.3 \
    --ppo-epsilon 0.01 \
    --ppo-max-mean-actor-kl 1e-3 \
    --optimizer-burnin-epochs 8 \
    --value-burnin-epochs 8 \
    --learner-param-dtype float32 \
    --length 128 \
    --warmup-updates 0 \
    --updates 100 \
    --save-path "$OUTDIR/latest.pkl" \
    --save-every 10 \
    --log-jsonl "$OUTDIR/metrics.jsonl" \
    --print-every 1 \
    "$@"
