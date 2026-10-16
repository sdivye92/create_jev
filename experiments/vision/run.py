"""One experiment on the synthetic shapes images, with the image controls. Example: python experiments/vision/run.py --tag main_crossentropy_seed0 --seed 0 --test"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402

if __name__ == "__main__":
    engine.run(engine.add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args(), "shapes")
