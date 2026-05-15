#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC="${SRC:-/mnt/gigahome/git/slippi-ai/experiments/doubles_delay_21_all_v19/latest.pkl}"
DST_DIR="${DST_DIR:-${REPO_DIR}}"
STATE_FILE="${STATE_FILE:-${DST_DIR}/.currentmodel_index}"
START_INDEX="${START_INDEX:-3}"

if [[ ! -f "${SRC}" ]]; then
  echo "Source file not found: ${SRC}" >&2
  exit 1
fi

if [[ ! -d "${DST_DIR}" ]]; then
  echo "Destination directory not found: ${DST_DIR}" >&2
  exit 1
fi

if [[ ! -f "${STATE_FILE}" ]]; then
  printf '%s\n' "${START_INDEX}" > "${STATE_FILE}"
fi

index="$(<"${STATE_FILE}")"
if [[ ! "${index}" =~ ^[0-9]+$ ]]; then
  echo "State file must contain a non-negative integer: ${STATE_FILE}" >&2
  exit 1
fi

target="${DST_DIR}/currentmodel${index}.pkl"
cp "${SRC}" "${target}"
printf '%s\n' "$((index + 1))" > "${STATE_FILE}"

echo "Copied ${SRC} -> ${target}; next index: $((index + 1))"
