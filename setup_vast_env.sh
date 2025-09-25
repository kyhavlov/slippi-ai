#!/bin/bash

# Install dependencies 
sudo apt install -y git libopengl0 libfuse2 qt6-qpa-plugins libasound2 libgl1 libusb-1.0-0-dev net-tools \
    python-is-python3 python3-pip python3.11-venv xvfb fuse
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../libmelee/
./Slippi_Online-x86_64.AppImage --appimage-extract
./squashfs-root/usr/bin/dolphin-emu --version
