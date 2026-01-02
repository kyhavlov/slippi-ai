# Shared-Memory Fused-Ports Inference Plan (RL Throughput)

This note captures the agreed plan for improving RL throughput by reducing IPC
overhead and better decoupling CPU work (env + preprocessing) from GPU work
(policy inference), inspired by the HAL project’s shared-memory evaluation loop.

## Target

- **Fast path** targets **`fuse_ports_inference=True`** (port-major `P*B` batch).
- Shared memory stores **final model input tensors** (the dtype/shape arrays fed
  to the compiled TF policy), not raw libmelee objects or Python-heavy `EnvOutput`
  structures.

## Status (2026-01-01)

- Implemented a **minimal-risk shared-memory transport for async env outputs**:
  `AsyncBatchedEnvironmentMP(use_shared_memory=True, shm_depth=...)` now uses
  `AsyncEnvShmMP` to stream `EnvOutput` leaves through a single shared-memory
  segment per env subprocess, while still using the Pipe for tiny control
  messages.
- Wired behind `--config.actor.env_output_shm=True` (RL only; requires
  `--config.actor.async_envs=True`).

This immediately removes the biggest source of overhead when items are enabled:
shipping a large nested object graph across process boundaries via pickling.

Important: `shm_depth` must be large enough to avoid ring-wrap while the main
process still holds references to previous states during a rollout. For RL we
default to `rollout_length + 2`, but it can be overridden via
`--config.actor.env_output_shm_depth=...`.

## Why

Current bottlenecks include:

- Large Python object graphs (nested namedtuples + many small numpy arrays) moving
  across process boundaries via `multiprocessing.Pipe`/pickling.
- TensorFlow overhead converting many small numpy arrays into `tf.Tensor`s at
  the Python boundary (especially visible when items are enabled).
- Underutilization from CPU↔GPU coupling; stalls in env/preprocess block GPU
  batching and vice versa.

## Design (minimal-risk version first)

### Data format

- Define a stable “packed input” layout for fused inference, with batch dimension
  `P*B` (ports are contiguous blocks of `B`).
- Allocate shared memory for the input fields as **contiguous numpy arrays**
  matching the policy’s expected dtype/shape.

### Producer/consumer

- **Producer** (eventually: Dolphin env worker; initially: fake generator):
  - Writes the next timestep’s fused-port inputs into shared memory (and
    `needs_reset`).
  - Sends a tiny “ready” header (e.g. `seq`/slot id) over a pipe/queue.
- **Consumer** (policy inference):
  - Reads the ready header with a hard timeout (never hang forever).
  - Reads inputs from shared memory and runs the compiled TF policy.
  - (Optional) Maintains the multi-frame context window locally (HAL-style)
    to avoid re-sending K frames per step.

### Synchronization & robustness

- Prefer tiny pipe messages (or a `Queue`) over per-env `Event`s at large scale.
- Any “wait for ready” path must have a **hard timeout** and trigger restart of
  the stuck producer rather than hanging indefinitely.
- Shared memory segments must be:
  - sized conservatively for local testing
  - cleaned up (`close()` and `unlink()`) on shutdown

## Rollout storage (follow-up)

The first implementation targets inference throughput. A later step can extend
the approach to trajectory storage to avoid extra per-step allocations/copies,
but it is more invasive and should be done after the inference path is stable.

## Next steps (fully packed model inputs)

The implemented `EnvOutput` shm transport is a big win, but it still leaves
Python-side TF input packing work in the main process.

Follow-up plan:

1. In env subprocesses, generate **already-embedded** `Game` inputs and write
   **packed-by-dtype** vectors into shm (so the TF boundary sees only a few big
   tensors, even with items).
2. Use `PackingPlan.pack_into(...)` to fill preallocated shm-backed arrays
   without per-step allocations.
3. Keep inference+learner in the main process (single GPU), and keep actions on
   the Pipe initially (small messages).

## Testing path (conservative)

1. Implement a **no-Dolphin** benchmark harness:
   - fake producer writes correctly-shaped inputs into shared memory
   - consumer runs compiled policy and measures fps / agent_step
   - strict runtime limits; confirm no leaked shared memory segments
2. Integrate shared-memory inputs into the **fake-env RL path** to validate
   correctness of the RL loop without Dolphin complexity.
3. Only then integrate with real Dolphin env workers and re-benchmark end-to-end.
