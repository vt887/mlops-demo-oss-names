# -*- coding: utf-8 -*-

import argparse
import configparser
import socket
import sys
from pathlib import Path

import torch

# Ensure project root is available for absolute imports when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rnn.rnn_model import RNN, all_letters, line_to_tensor, n_letters


def load_config():
    config = configparser.ConfigParser()
    config.read(PROJECT_ROOT / "config" / "config.ini")
    base_dump_dir = Path(config["DEFAULT"]["dump_dir"])
    hostname = socket.gethostname()
    return {
        "data_dir": Path(config["DEFAULT"]["data_dir"]),
        "data_files": config["DEFAULT"]["data_files"],
        "dump_dir": base_dump_dir,
        "dump_dir_with_host": base_dump_dir / hostname,
        "output_file": config["DEFAULT"]["output_file"],
    }


def load_categories(data_dir: Path, pattern: str):
    category_files = sorted(data_dir.glob(pattern))
    categories = [file.stem for file in category_files]
    if not categories:
        raise RuntimeError(f"No category files found in {data_dir} (pattern: {pattern}).")
    return categories


def load_model(n_categories: int, checkpoint: Path, fallback: Path) -> RNN:
    model = RNN(n_letters, 128, n_categories)
    if not checkpoint.exists():
        if fallback.exists():
            checkpoint = fallback
        else:
            raise FileNotFoundError(
                f"Checkpoint {checkpoint} not found and no fallback at {fallback}. Run training first."
            )
    state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: RNN, categories, name: str, n_predictions: int):
    cleaned = ''.join([c for c in name if c in all_letters])
    if cleaned == "":
        raise ValueError("Provided name has no valid characters.")

    with torch.no_grad():
        hidden = model.init_hidden()
        line_tensor = line_to_tensor(cleaned)
        output = None
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
        probs = torch.exp(output)
        topv, topi = probs.topk(min(n_predictions, probs.size(1)), dim=1)

    return [
        {
            "category": categories[topi[0][i].item()],
            "probability": topv[0][i].item(),
        }
        for i in range(topv.size(1))
    ]


def main():
    parser = argparse.ArgumentParser(description="Predict the category of a given name.")
    parser.add_argument("name", help="Name to classify")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to display")
    args = parser.parse_args()

    cfg = load_config()
    categories = load_categories(cfg["data_dir"], cfg["data_files"])
    checkpoint_path = cfg["dump_dir_with_host"] / cfg["output_file"]
    fallback_path = cfg["dump_dir"] / cfg["output_file"]
    model = load_model(len(categories), checkpoint_path, fallback_path)

    predictions = predict(model, categories, args.name, args.top)

    for rank, item in enumerate(predictions, start=1):
        probability = item["probability"] * 100
        print(f"{rank}. {item['category']} ({probability:.2f}%)")


if __name__ == "__main__":
    main()
