# -*- coding: utf-8 -*-

# Import necessary libraries
from fastapi import FastAPI
import torch
from .rnn_model import RNN, all_letters, n_letters, letter_to_tensor, line_to_tensor
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
dump_dir = Path(config["DEFAULT"]["dump_dir"])
output_file = config["DEFAULT"]["output_file"]

# Use the machine hostname for organizing outputs
hostname = socket.gethostname()
dump_dir = dump_dir / hostname

# Dynamically load categories from the training data
def load_categories():
    # Find all category files in the data directory
    category_files = list(data_dir.glob(data_files))
    return [file.stem for file in category_files]

categories = load_categories()
n_categories = len(categories)

# Load the trained RNN model
model = RNN(n_letters, 128, n_categories)
model.load_state_dict(torch.load(dump_dir / output_file))
model.eval()

# Log category-to-index mapping for debugging
print("Category-to-index mapping:", {i: category for i, category in enumerate(categories)})

@app.get("/predict")
def predict(name: str):
    """
    Predict the category of a given name.

    Args:
        name (str): The name to classify.

    Returns:
        dict: A dictionary containing the name and predicted category.
    """
    with torch.no_grad():
        # Initialize the hidden state
        hidden = model.init_hidden()

        # Log raw output probabilities for debugging
        output, _ = model(letter_to_tensor(name), hidden)
        print("Raw output probabilities:", output.tolist())

        # Get the predicted category index
        category_index = output.argmax().item()

        # Return the name and predicted category
        return {"name": name, "predicted_category": categories[category_index]}
