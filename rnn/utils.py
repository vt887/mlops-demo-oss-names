import os
import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model for Names classification with a Character-Level RNN.")
    parser.add_argument("--dump_dir", type=Path, default="results", help="Dump path.")
    parser.add_argument("--experiment_name", type=str, default=os.uname()[1], help="Experiment name, defaults to job ID.")
    parser.add_argument("--data_path", type=Path, default="data", help="Path to folder, where training data is located")
    parser.add_argument("--n_iters", type=int, default=100000, help="Number of iterations to train the model")

    return parser.parse_args()
