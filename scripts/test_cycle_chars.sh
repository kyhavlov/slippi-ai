#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# .config/Slippi Launcher/netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage
# /home/kyle/.config/Slippi Launcher/netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage
# /home/kyle/Downloads/Slippi_Online-x86_64.AppImage
# Slippi_Online-x86_64-ExiAI.AppImage
export ROOTDIR=$(pwd)

python scripts/test_cycle_env_chars.py \
  --config.dolphin.path="/home/kyle/ray-resources/Slippi_Netplay_Mainline_NoGui-x86_64.AppImage" \
  --config.dolphin.iso="/home/kyle/ray-resources/SSBM.iso" \
  --config.dolphin.headless=True \
  --config.dolphin.save_replays=True \
  --config.dolphin.console_timeout=10 \
  --config.dolphin.disable_audio=True \
  --config.dolphin.infinite_time=False \
  --config.dolphin.blocking_input=True \
  --config.dolphin.emulation_speed=0 \
  --config.agent.path="$ROOTDIR/experiments/doubles_delay_18_all_v2/1000k_steps.pkl" \
  --config.agent.name="Ralph,Tempo,Darkatma,xRunRiot,Woopty,MisterGW" \
  --config.actor.num_env_steps=4 \
  --config.actor.inner_batch_size=1 \
  --config.dolphin.log_types="SLIPPI_ONLINE,SLIPPI" \
  --config.dolphin.log_level=3
