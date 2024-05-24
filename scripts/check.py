import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-ps", "--per_step", default=-1, type=int, required=False)


args = parser.parse_args()
print(args.per_step)

