#!/usr/bin/env sh
set -eu

OUTDIR="reports/triage/jax_sim_rl_runs/imitation_v17_sim_foxditto_lr3e6_kl5e2_pb1_w32x256_eval1"
mkdir -p "$OUTDIR"

exec env -u LD_LIBRARY_PATH \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  MELEE_SIM_DATA=/home/kyle/git/melee-sim-light-alt/data \
  PYTHONPATH=. \
  .venv/bin/python scripts/benchmark_jax_sim_rl.py \
    --model-path models/imitation_v17_2000k.pkl \
    --workers 32 \
    --batch-size 256 \
    --matchup fox-fox \
    --rollout-length 80 \
    --ppo-batches 1 \
    --ppo-epochs 1 \
    --learner-minibatch-size 1024 \
    --learner-minibatch-scan-size 32 \
    --learning-rate 3e-6 \
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
    --ppo-max-mean-actor-kl 1e-4 \
    --revert-on-post-update-actor-kl \
    --post-update-eval-interval 1 \
    --optimizer-burnin-epochs 8 \
    --value-burnin-epochs 8 \
    --learner-param-dtype float32 \
    --length 128 \
    --initial-stagger-steps 600 \
    --warmup-updates 0 \
    --updates 1000 \
    --save-path "$OUTDIR/latest.pkl" \
    --save-every 50 \
    --log-jsonl "$OUTDIR/metrics.jsonl" \
    --print-every 1 \
    "$@"
