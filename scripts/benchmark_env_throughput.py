import time

from absl import app
from absl import flags

import numpy as np

from slippi_ai import dolphin
from slippi_ai import envs as env_lib
from slippi_ai.types import Controller, Stick, Buttons


FLAGS = flags.FLAGS

flags.DEFINE_string('dolphin_path', None, 'Path to dolphin executable/AppImage.', required=True)
flags.DEFINE_string('iso_path', None, 'Path to SSBM iso.', required=True)
flags.DEFINE_integer('num_envs', 8, 'Number of parallel environments.')
flags.DEFINE_integer('inner_batch_size', 1, 'Number of envs per env subprocess.')
flags.DEFINE_integer('num_steps', 0, 'Time-batching factor for AsyncBatchedEnvironmentMP (0 disables).')
flags.DEFINE_integer('frames', 1800, 'Number of in-game frames to step.')
flags.DEFINE_integer('online_delay', 0, 'Online delay.')
flags.DEFINE_boolean('headless', True, 'Use headless mode (EXI+FFW when supported).')
flags.DEFINE_boolean('disable_audio', True, 'Disable audio.')
flags.DEFINE_float('emulation_speed', 0.0, 'Mainline-only; set 0 for unlimited.')
flags.DEFINE_boolean('infinite_time', True, 'Infinite time / no stocks.')
flags.DEFINE_integer('console_timeout', 30, 'Seconds before console timeout.')
flags.DEFINE_integer('log_level', 0, 'Dolphin log level (0 to disable).')
flags.DEFINE_boolean('include_controller_state', True, 'Include controller state in observations.')


def _neutral_controllers(batch_size: int):
  stick = Stick(
      x=np.full([batch_size], 0.5, dtype=np.float32),
      y=np.full([batch_size], 0.5, dtype=np.float32),
  )
  buttons = Buttons(
      A=np.full([batch_size], False, dtype=np.bool_),
      B=np.full([batch_size], False, dtype=np.bool_),
      X=np.full([batch_size], False, dtype=np.bool_),
      Y=np.full([batch_size], False, dtype=np.bool_),
      Z=np.full([batch_size], False, dtype=np.bool_),
      L=np.full([batch_size], False, dtype=np.bool_),
      R=np.full([batch_size], False, dtype=np.bool_),
      D_UP=np.full([batch_size], False, dtype=np.bool_),
  )
  controller = Controller(
      main_stick=stick,
      c_stick=stick,
      shoulder=np.full([batch_size], 0.0, dtype=np.float32),
      buttons=buttons,
  )
  return {port: controller for port in (1, 2, 3, 4)}


def main(_):
  num_envs = int(FLAGS.num_envs)
  inner_batch_size = int(FLAGS.inner_batch_size)
  num_steps = int(FLAGS.num_steps)
  frames = int(FLAGS.frames)

  dolphin_kwargs = dict(
      path=FLAGS.dolphin_path,
      iso=FLAGS.iso_path,
      players={port: dolphin.AI() for port in (1, 2, 3, 4)},
      online_delay=int(FLAGS.online_delay),
      headless=bool(FLAGS.headless),
      disable_audio=bool(FLAGS.disable_audio),
      emulation_speed=float(FLAGS.emulation_speed),
      infinite_time=bool(FLAGS.infinite_time),
      console_timeout=float(FLAGS.console_timeout),
      log_level=int(FLAGS.log_level),
      save_replays=False,
  )

  env = env_lib.AsyncBatchedEnvironmentMP(
      num_envs=num_envs,
      dolphin_kwargs=dolphin_kwargs,
      num_steps=num_steps,
      inner_batch_size=inner_batch_size,
      swap_ports=False,
      agent_names=[('MP', 'MP', 'MP', 'MP')] * num_envs,
      env_ids=list(range(num_envs)),
      scheduler=None,
      enable_singles=False,
      include_controller_state=bool(FLAGS.include_controller_state),
  )

  controllers = _neutral_controllers(num_envs)

  try:
    # Prime the pipeline: there's always an initial state available.
    env.pop()

    start = time.perf_counter()
    if num_steps == 0:
      for _ in range(frames):
        env.push(controllers)
        env.pop()
    else:
      num_chunks, rem = divmod(frames, num_steps)
      if rem:
        raise ValueError('--frames must be divisible by --num_steps when --num_steps > 0')
      for _ in range(num_chunks):
        for _ in range(num_steps):
          env.push(controllers)
        for _ in range(num_steps):
          env.pop()
    dt = time.perf_counter() - start

    env_fps = (frames * num_envs) / dt
    print(
        f'num_envs={num_envs} inner_batch_size={inner_batch_size} num_steps={num_steps} '
        f'frames={frames} seconds={dt:.3f} env_fps={env_fps:.1f}'
    )
  finally:
    env.stop()


if __name__ == '__main__':
  app.run(main)
