# -*- coding: utf-8 -*-

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import configparser
import socket
from rnn.rnn_model import RNN, letter_to_tensor, line_to_tensor
import random
import time
import math

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
    """
    Load category names and their corresponding lines from the configured data directory.

    Returns:
        tuple[list[str], dict[str, list[str]]]: Ordered categories and per-category line samples.
    """
    category_files = list(data_dir.glob(data_files))
    categories = [file.stem for file in category_files]
    category_lines = {}
    for file in category_files:
        with open(file, encoding="utf-8") as f:
            category_lines[file.stem] = f.read().strip().split("\n")
    return categories, category_lines

categories, category_lines = load_categories()
n_categories = len(categories)

# Randomly select a training example
def random_training_example():
    """
    Sample a random (category, line) pair converted into tensors for training.

    Returns:
        tuple[str, str, torch.Tensor, torch.Tensor]:
            Category name, raw line, category tensor, and encoded line tensor.
    """
    category = random.choice(categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

# Initialize the RNN model
n_hidden = 128
model = RNN(n_letters, n_hidden, n_categories)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Helper function to calculate time elapsed
def time_since(since):
    """
    Format elapsed wall-clock time since the provided timestamp.

    Args:
        since (float): Epoch timestamp captured before training started.

    Returns:
        str: Human-readable minutes and seconds string.
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Train the model on a single example
def train(category_tensor, line_tensor):
    """
    Execute a single training step for one name/category pair.

    Args:
        category_tensor (torch.Tensor): Target category indices.
        line_tensor (torch.Tensor): Encoded character sequence for the sampled name.

    Returns:
        tuple[torch.Tensor, float]: Model output logits and the scalar loss value.
    """
    hidden = model.init_hidden()

    model.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

# Training loop
n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []

start = time.time()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=dump_dir)

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print progress
    if iter % print_every == 0:
        guess = categories[output.argmax().item()]
        correct = "✓" if guess == category else f"✗ ({category})"
        print(f"{iter} {iter / n_iters * 100:.2f}% ({time_since(start)}) {loss:.4f} {line} / {guess} {correct}")

    # Add loss to TensorBoard
    if iter % plot_every == 0:
        writer.add_scalar("Loss/train", current_loss / plot_every, iter)
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Save the trained model
torch.save(model.state_dict(), dump_dir / output_file)
print(f"Model saved to {dump_dir / output_file}")

# Close TensorBoard writer
writer.close()
