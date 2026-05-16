import argparse

from slippi_ai.jax import checkpoint_conversion


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  parser.add_argument('output')
  parser.add_argument(
      '--learner-param-dtype',
      choices=('float32', 'bfloat16'),
      default='float32',
  )
  args = parser.parse_args()

  checkpoint_conversion.convert_file(
      args.source,
      args.output,
      learner_param_dtype=args.learner_param_dtype,
  )


if __name__ == '__main__':
  main()
