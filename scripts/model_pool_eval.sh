#!/bin/bash

# Script to run model pool evaluation

PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH
ROOTDIR=$(pwd)
export ROOTDIR
TF_CPP_MIN_LOG_LEVEL=3
export TF_CPP_MIN_LOG_LEVEL

# Default paths - modify these as needed
DOLPHIN_PATH="$ROOTDIR/Slippi_Netplay_Mainline_NoGui-x86_64.AppImage"
ISO_PATH="$ROOTDIR/SSBM.iso"
OUTPUT_DIR="$ROOTDIR/eval/model_pool_results"
MODEL_DIR="$ROOTDIR/eval/models"
MODEL_PATTERN="*.pkl"
DOLPHIN_HEADLESS=True

# Default parameters
MAX_PARALLEL_GAMES=2
DISPLAY_INTERVAL=5
RANDOMIZE_CHARS=True
NOVELTY_WEIGHT=50.0
SKILL_BIAS_WEIGHT=25.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir=*)
      MODEL_DIR="${1#*=}"
      shift
      ;;
    --model-pattern=*)
      MODEL_PATTERN="${1#*=}"
      shift
      ;;
    --dolphin=*)
      DOLPHIN_PATH="${1#*=}"
      shift
      ;;
    --iso=*)
      ISO_PATH="${1#*=}"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --parallel=*)
      MAX_PARALLEL_GAMES="${1#*=}"
      shift
      ;;
    --interval=*)
      DISPLAY_INTERVAL="${1#*=}"
      shift
      ;;
    --randomize-chars=*)
      RANDOMIZE_CHARS="${1#*=}"
      shift
      ;;
    --headless=*)
      DOLPHIN_HEADLESS="${1#*=}"
      shift
      ;;
    --novelty-weight=*)
      NOVELTY_WEIGHT="${1#*=}"
      shift
      ;;
    --skill-bias=*)
      SKILL_BIAS_WEIGHT="${1#*=}"
      shift
      ;;
    *)
      # Unknown option
      echo "Unknown option: $1"
      echo "Usage: $0 [--model-dir=/path/to/models] [--model-pattern='*.pkl'] [--dolphin=/path/to/dolphin] [--iso=/path/to/iso] [--output-dir=/path/to/output] [--parallel=2] [--interval=10] [--randomize-chars=True] [--headless=True] [--novelty-weight=5.0] [--skill-bias=0.0]"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"

echo "Starting model pool evaluation with the following settings:"
echo "  - Model directory: $MODEL_DIR"
echo "  - Model pattern: $MODEL_PATTERN"
echo "  - Dolphin path: $DOLPHIN_PATH"
echo "  - ISO path: $ISO_PATH"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Parallel games: $MAX_PARALLEL_GAMES"
echo "  - Display interval: $DISPLAY_INTERVAL"
echo "  - Randomize characters: $RANDOMIZE_CHARS"
echo "  - Headless mode: $DOLPHIN_HEADLESS"
echo "  - Novelty weight: $NOVELTY_WEIGHT"
echo "  - Skill bias weight: $SKILL_BIAS_WEIGHT"

# Run the evaluation script
python scripts/model_pool_eval.py \
  --dolphin_path="$DOLPHIN_PATH" \
  --dolphin_iso="$ISO_PATH" \
  --dolphin_headless="$DOLPHIN_HEADLESS" \
  --model_dir="$MODEL_DIR" \
  --model_pattern="$MODEL_PATTERN" \
  --max_parallel_games="$MAX_PARALLEL_GAMES" \
  --output_dir="$OUTPUT_DIR" \
  --display_interval="$DISPLAY_INTERVAL" \
  --randomize_characters="$RANDOMIZE_CHARS" \
  --novelty_weight="$NOVELTY_WEIGHT" \
  --skill_bias_weight="$SKILL_BIAS_WEIGHT"
