#!/usr/bin/env sh

# These parameters are optimized for an i7-11700K and RTX 3080Ti with 64GB of RAM.
# I suggest increasing the num_envs until you run out of RAM, and then increasing
# the rollout_length until you run out of GPU memory. The inner_batch_size should
# be set so that num_envs / inner_batch_size is approximately the number of CPU
# threads you have available. The rest of the parameters can be left as is.

# Slippi_Online-x86_64-ExiAI.AppImage
#

# What player(s) from the dataset should we condition on?
# This can be a comma-separated list.
NAME="Master Player,Ralph"
D=18
TAG=rl_doubles_delay_${D}_v2
ROOTDIR=$(pwd)
#DOLPHIN_PATH="$ROOTDIR/Slippi_Online-x86_64-ExiAI.AppImage"
DOLPHIN_PATH="$ROOTDIR/Slippi_Netplay_Mainline_NoGui-x86_64.AppImage"
ISO_PATH="$ROOTDIR/SSBM.iso"

export PYTHONPATH="."

python slippi_ai/rl/run.py \
  --config.runtime.tag=$TAG \
  --config.runtime.max_step=10000 \
  --config.runtime.log_interval=300 \
  --config.dolphin.path="$DOLPHIN_PATH" \
  --config.dolphin.iso="$ISO_PATH" \
  --config.dolphin.headless=False \
  --config.dolphin.console_timeout=60 \
  --config.dolphin.infinite_time=False \
  --config.dolphin.disable_audio=True \
  --config.dolphin.emulation_speed=0 \
  --config.learner.learning_rate=3e-5 \
  --config.learner.value_cost=1 \
  --config.learner.reward_halflife=4 \
  --config.learner.reward.damage_ratio=0.00333 \
  --config.learner.reward.ledge_grab_penalty=0.01 \
  --config.learner.policy_gradient_weight=5 \
  --config.learner.kl_teacher_weight=3e-3 \
  --config.learner.ppo.num_epochs=2 \
  --config.learner.ppo.num_batches=16 \
  --config.learner.ppo.beta=3e-1 \
  --config.learner.ppo.epsilon=1e-2 \
  --config.learner.ppo.minibatched=False \
  --config.teacher="$ROOTDIR/models/latest_inclsingles_4750k.pkl" \
  --config.opponent.type=self \
  --config.opponent.train=True \
  --config.actor.rollout_length=300 \
  --config.actor.num_envs=4 \
  --config.actor.inner_batch_size=2 \
  --config.actor.async_envs=True \
  --config.actor.num_env_steps=0 \
  --config.actor.gpu_inference=True \
  --config.actor.enable_singles=False \
  --config.agent.name="$NAME" \
  --config.agent.batch_steps=4 \
  --config.runtime.reset_every_n_steps=512 \
  --config.runtime.burnin_steps_after_reset=1 \
  --config.optimizer_burnin_steps=0 \
  --config.value_burnin_steps=0 \
  --wandb.name=$TAG \
  --wandb.mode=disabled \
  --wandb.tags=ppo \
  "$@"
