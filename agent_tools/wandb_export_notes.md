W&B export notes (2025-11-20)
--------------------------------
- API auth: `WANDB_API_KEY` stored in `wandb_key.txt`; export with `export WANDB_API_KEY=$(cat wandb_key.txt)`.
- Project: `kyhavlov-personal/slippi-ai`.
- For tag `rl_doubles_delay_21_v18`, exported run histories to `wandb_exports_v18/`:
  - Run IDs with data: `2z79rdtb`, `6y576vxf`, `sasf5ixi`, `f05ffpz8` (others were empty).
  - Command pattern:
    ```bash
    python - <<'PY'
    import pathlib, wandb, pandas as pd
    api = wandb.Api()
    OUT = pathlib.Path("wandb_exports_v18"); OUT.mkdir(exist_ok=True)
    for run_id in ["2z79rdtb","6y576vxf","sasf5ixi","f05ffpz8"]:
        r = api.run("kyhavlov-personal/slippi-ai/" + run_id)
        df = r.history(keys=None, samples=None, pandas=True)
        df.insert(0, "run_name", r.name)
        df.to_csv(OUT / f"{r.name}_{r.id}.csv", index=False)
    PY
    ```
- For comparison tag `rl_doubles_delay_18_v5`, non-empty runs exported to `wandb_exports_18_v5/`:
  - Run IDs with data: `4xgbamt3`, `2devuk70`, `isqky41o`, `13k01ny2` (others were empty).
- Notes:
  - Wandb “Unsafe field attributes” warnings from pydantic are harmless.
  - Histories include full keys; summaries/configs omitted to avoid serialization issues. Add if needed.

