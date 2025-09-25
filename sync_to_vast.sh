#!/bin/bash

SSH_PORT=22
SSH_HOST=kyle@34.102.73.49
REMOTE_DIR=/home/kyle

rsync -zarv -e "ssh -p $SSH_PORT" --exclude="*/__pycache__" --include="*.py" --include="*/" --exclude="*" slippi_ai $SSH_HOST:$REMOTE_DIR/slippi-ai/
rsync -zarv -e "ssh -p $SSH_PORT" --exclude="*/__pycache__" --include="*.py" --include="*/" --exclude="*" slippi_db $SSH_HOST:$REMOTE_DIR/slippi-ai/
rsync -zarv -e "ssh -p $SSH_PORT" --exclude="*/__pycache__" --include="*.py" --include="*.sh" --include="*/" --exclude="*" scripts $SSH_HOST:$REMOTE_DIR/slippi-ai/
rsync -zarv -e "ssh -p $SSH_PORT" --exclude="*/__pycache__" --exclude=".venv/*" --exclude="build/*" ../libmelee/ $SSH_HOST:$REMOTE_DIR/libmelee/
rsync -zarv -e "ssh -p $SSH_PORT" setup_vast_env.sh $SSH_HOST:$REMOTE_DIR/slippi-ai/
rsync -zarv -e "ssh -p $SSH_PORT" bot3-user.json requirements.txt Slippi_Online-x86_64.AppImage SSBM.iso $SSH_HOST:$REMOTE_DIR/slippi-ai/
rsync -zarv -e "ssh -p $SSH_PORT" models/rl_doubles_v5_3400.pkl $SSH_HOST:$REMOTE_DIR/slippi-ai/models/
