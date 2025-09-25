#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# .config/Slippi Launcher/netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage
# /home/kyle/.config/Slippi Launcher/netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage
# /home/kyle/Downloads/Slippi_Online-x86_64.AppImage
# Slippi_Netplay_Mainline_NoGui-x86_64.AppImage
# Slippi_Online-x86_64-ExiAI.AppImage
export ROOTDIR=$(pwd)

python scripts/test_ffw.py \
  --dolphin.path="$ROOTDIR/Slippi_Online-x86_64-ExiAI.AppImage" \
  --dolphin.iso="$ROOTDIR/SSBM.iso" \
  --dolphin.replay_dir="$ROOTDIR/bot-replays/" \
  --dolphin.infinite_time=False \
  --debug=True
