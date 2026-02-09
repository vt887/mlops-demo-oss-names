# -*- coding: utf-8 -*-

# Import necessary libraries
from fastapi import FastAPI
import torch
from .rnn_model import RNN, n_letters, line_to_tensor, all_letters
from pathlib import Path
import configparser
import socket

# Initialize FastAPI application
app = FastAPI()

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config/config.ini")

# Retrieve parameters from config.ini
data_dir = Path(config["DEFAULT"]["data_dir"])
data_files = config["DEFAULT"]["data_files"]
base_dump_dir = Path(config["DEFAULT"]["dump_dir"])
output_file = config["DEFAULT"]["output_file"]

# Use the machine hostname for organizing outputs (some checkpoints store under results/<hostname>/)
hostname = socket.gethostname()
dump_dir_with_host = base_dump_dir / hostname

# Dynamically load categories from the training data
def load_categories():
    # Find all category files in the data directory
    category_files = list(data_dir.glob(data_files))
    return [file.stem for file in category_files]

categories = load_categories()
n_categories = len(categories)

# Validate categories
if n_categories == 0:
    raise RuntimeError(f"No category files found in {data_dir} (pattern: {data_files}). Check your config and working directory.")

# Load the trained RNN model
model = RNN(n_letters, 128, n_categories)
model.load_state_dict(torch.load(dump_dir / output_file))
model.eval()

# Log category-to-index mapping for debugging
print("Category-to-index mapping:", {i: category for i, category in enumerate(categories)})

@app.get("/predict")
def predict(name: str, n_predictions: int = 1):
    """
    Predict the category of a given name.

    Args:
        name (str): The name to classify.
        n_predictions (int): Number of top predictions to return.

    Returns:
        dict: A dictionary containing the name, predicted_category (top-1), and a list of top predictions with probabilities.
    """
    if not isinstance(name, str) or name.strip() == "":
        return {"error": "Invalid or empty name provided."}

    # Clean the name by keeping only characters known to the model
    cleaned = ''.join([c for c in name if c in all_letters])
    if cleaned == "":
        return {"error": "Name contains no valid characters after filtering."}

    with torch.no_grad():
        # Initialize the hidden state
        hidden = model.init_hidden()

        # Convert full name into a sequence tensor and run through the RNN one char at a time
        line_tensor = line_to_tensor(cleaned)
        output = None
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)

        # Output is log-probabilities (LogSoftmax). Convert to probabilities for clearer API values
        probs = torch.exp(output)

        # Get top N predictions
        topv, topi = probs.topk(min(n_predictions, probs.size(1)), dim=1)

        predictions = []
        for i in range(topv.size(1)):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append({"category": categories[category_index], "probability": value})

        # Top-1 predicted category for backwards compatibility with tests
        predicted_category = categories[topi[0][0].item()]

        return {"name": name, "predicted_category": predicted_category, "predictions": predictions}
