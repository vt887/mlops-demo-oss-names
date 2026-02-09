# -*- coding: utf-8 -*-

# Import necessary libraries
import torch
import torch.nn as nn
import string

# Define the set of all possible letters and the input layer size
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Convert a single letter to a one-hot encoded tensor
def letter_to_tensor(letter):
    """
    Convert a single letter to a one-hot encoded tensor.

    Args:
        letter (str): A single character.

    Returns:
        torch.Tensor: A tensor of shape (1, n_letters) with one-hot encoding.
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.find(letter)] = 1
    return tensor

# Convert a line (string) to a tensor of one-hot encoded letters
def line_to_tensor(line):
    """
    Convert a string to a tensor of one-hot encoded letters.

    Args:
        line (str): A string of characters.

    Returns:
        torch.Tensor: A tensor of shape (line_length, 1, n_letters).
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# Define the RNN class
class RNN(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) for character-level classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN model.

        Args:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # Define layers for input-to-hidden and input-to-output transformations
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        Perform a forward pass through the RNN.

        Args:
            input (torch.Tensor): The input tensor of shape (1, input_size).
            hidden (torch.Tensor): The hidden state tensor of shape (1, hidden_size).

        Returns:
            tuple: The output tensor and the next hidden state tensor.
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state to zeros.

        Returns:
            torch.Tensor: A tensor of shape (1, hidden_size) initialized to zeros.
        """
        return torch.zeros(1, self.hidden_size)
