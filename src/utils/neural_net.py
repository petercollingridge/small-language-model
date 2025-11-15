import torch

import torch.optim as optim
import torch.nn.functional as F
from torch import nn


def get_model(input_size, hidden_size, output_size):
    """ Create a simple neural network model. """

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Linear(hidden_size, output_size),
    )
    return model


def load_model(filepath):
    """
    Load a trained model from a file.
    """

    print(f"Loading model from {filepath}...")
    model_state = torch.load(filepath)

    print(model_state)

    return model_state


def train_simple_model(input_vectors, output_vectors, hidden_size = 2, loop_n=10000):
    """ Train a simple model with a single hidden layer. """

    X = torch.FloatTensor(input_vectors)
    y = torch.FloatTensor(output_vectors)

    # Prepare layer sizes
    input_size = len(input_vectors[0])
    output_size = len(output_vectors[0])

    # Build the model
    model = get_model(input_size, hidden_size, output_size)

    # In Pytorch, Softmax is already added when using CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(loop_n):
        # Compute forward pass and print loss
        y_pred = model(X)

        loss = loss_fn(y_pred, torch.argmax(y, dim=1))

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights using gradient descent
        optimizer.step()

    return model


def run_model(model, input_vectors, threshold=0.05):
    """ Run the model on the input vectors. """

    X = torch.FloatTensor(input_vectors)
    y_pred = model(X)

    probabilties = torch.softmax(y_pred, dim=-1)

    # Zero out probabilities below the threshold
    filtered = torch.where(probabilties >= threshold, probabilties, torch.tensor(0.0, device=y_pred.device))

    # Renormalize to make sure the sum is still 1
    renormormalised = filtered / filtered.sum(dim=-1, keepdim=True)
    return renormormalised.detach().numpy()
