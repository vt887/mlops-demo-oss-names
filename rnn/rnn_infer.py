# -*- coding: utf-8 -*-

# Import necessary libraries
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from .rnn_model import RNN, n_letters, line_to_tensor, all_letters
from pathlib import Path
import configparser
import socket

# Initialize FastAPI application
app = FastAPI()


@app.get("/")
def root():
    """
    Present a friendly landing payload describing available endpoints.

    Returns:
        dict: Service metadata plus example routes.
    """
    return {
        "message": "Character-level name classifier service",
        "endpoints": {
            "health": "/health",
            "predict": "/predict?name=Ivanov&n_predictions=3",
        },
    }


@app.get("/health")
def health():
    """
    Report basic service liveness for external health checks.

    Returns:
        dict: Health payload containing a static OK status.
    """
    return {"status": "ok"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """
    Return a descriptive JSON body when no route matches the request.

    Args:
        request (Request): Incoming HTTP request information.
        exc (Exception): Raised routing exception.

    Returns:
        fastapi.responses.JSONResponse: JSON payload pointing to valid endpoints.
    """
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "path": request.url.path,
            "hint": "Use /predict?name=Ivanov or call /health for status",
        },
    )

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config/config.ini")

# Retrieve parameters from config.ini
data_dir = Path(config["DEFAULT"]["data_dir"])
data_files = config["DEFAULT"]["data_files"]
base_dump_dir = Path(config["DEFAULT"]["dump_dir"])
output_file = config["DEFAULT"]["output_file"]
n_predictions = config["DEFAULT"]["n_predictions"]

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
checkpoint_path = dump_dir_with_host / output_file
if not checkpoint_path.exists():
    fallback_path = base_dump_dir / output_file
    if fallback_path.exists():
        checkpoint_path = fallback_path
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {fallback_path}; ensure training has produced {output_file}.")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Log category-to-index mapping for debugging
print("Category-to-index mapping:", {i: category for i, category in enumerate(categories)})

@app.get("/predict")
def predict(name: str, n_predictions: int = n_predictions):
    """
    Predict the category of a given name.

    Args:
        name (str): The name to classify.
        n_predictions (int): Number of top predictions to return (default 3).

    Returns:
        dict: Response payload containing name, predicted_category, and predictions list.
    """
    if not isinstance(name, str) or name.strip() == "":
        return {"error": "Invalid or empty name provided."}

    # Guard against invalid n_predictions values to keep the API predictable
    if not isinstance(n_predictions, int) or n_predictions < 1:
        return {"error": "n_predictions must be a positive integer."}

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

        # Get top N predictions (limited to available categories)
        topv, topi = probs.topk(min(n_predictions, probs.size(1)), dim=1)

        predictions = []
        for i in range(topv.size(1)):
            value = float(topv[0][i].item())
            category_index = topi[0][i].item()
            predictions.append({"category": categories[category_index], "probability": value})

        # Top-1 predicted category for backwards compatibility with tests
        predicted_category = categories[topi[0][0].item()]

        return {"name": name, "predicted_category": predicted_category, "predictions": predictions}
