#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
  shift
fi

SRC_ROOT="${1:-/mnt/nvme0/projects/slippi-data/Raw}"
RAW_JSON="${2:-/mnt/gigahome/git/slippi-ai/data/raw.json}"
DST_ROOT="${3:-/mnt/gigahome/git/slippi-ai/data/Raw}"

RSYNC_FLAGS=(-av)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

SRC_ROOT="$SRC_ROOT" RAW_JSON="$RAW_JSON" python - <<'PY' | rsync "${RSYNC_FLAGS[@]}" --files-from=- "${SRC_ROOT}/" "${DST_ROOT}/"
import os, json, zipfile

raw_root = os.environ["SRC_ROOT"]
raw_json = os.environ["RAW_JSON"]

def load_rawjson(path):
    with open(path) as f:
        entries = json.load(f)
    return {e.get("name", "").replace('\\', '/') for e in entries if e.get("name")}

rawjson_names = load_rawjson(raw_json)


def relpath(p):
    return os.path.relpath(p, raw_root).replace('\\', '/')

needs = set()
for root, _, files in os.walk(raw_root):
    for fn in files:
        if fn.lower().endswith('.zip'):
            zp = os.path.join(root, fn)
            rel = relpath(zp)
            if rel not in rawjson_names:
                needs.add(rel)
            try:
                with zipfile.ZipFile(zp) as zf:
                    for name in zf.namelist():
                        lower = name.lower()
                        if lower.endswith('.slpz') or lower.endswith('.slpp'):
                            needs.add(rel)
                            break
            except zipfile.BadZipFile:
                needs.add(rel)

for rel in sorted(needs):
    print(rel)
PY
