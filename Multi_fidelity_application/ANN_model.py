import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress deprecation warnings (common with older PyTorch APIs)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import TensorDataset, DataLoader


class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds):
        """
        Initialize a simple Multi-Layer Perceptron (MLP).
        Args:
            input_dimension (int): Number of input features
            hidden_channels (int): Number of neurons in each hidden layer
            output_dimension (int): Number of output units
            n_hidden_layer (int): Number of layers
            activate (str): Activation function type ('relu', 'tanh', 'sigmoid')
            random_seeds (int): Random seed for reproducible weight initialization
        """
        super(MLP, self).__init__()

        # Set random seed for reproducible initialization
        torch.manual_seed(random_seeds)

        # Input layer → first hidden layer
        self.input_layer = nn.Linear(input_dimension, hidden_channels)

        # Intermediate hidden layers (if more than one hidden layer)
        # Stored in a ModuleList for proper parameter registration
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_hidden_layer - 1)
        ])

        # Final hidden layer → output layer
        self.output_layer = nn.Linear(hidden_channels, output_dimension)

        # Select activation function (defaults to tanh)
        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dimension)
        """
        # Input layer + activation
        x = self.act_fn(self.input_layer(x))

        # Pass through all intermediate hidden layers with activation
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x))

        # Output layer (no activation — typical for regression)
        output = self.output_layer(x)
        return output


def SingleANN_M(INPUT, OUTPUT, bili, n_layer, lr1, weight_decay1, Epoch1, hidden_channels1, activate, batch_size1,
                random_seeds1):
    """
    Train a single Artificial Neural Network (one experiment/run).

    Args:
        INPUT (torch.Tensor): Input features (n_samples, n_features)
        OUTPUT (torch.Tensor): Target values (n_samples, 1)
        bili (float): Proportion of data used for training (0.0–1.0)
        n_layer (int): Number of layers
        lr1 (float): Learning rate
        weight_decay1 (float): L2 regularization strength (weight decay)
        Epoch1 (int): Maximum number of training epochs
        hidden_channels1 (int): Number of neurons per hidden layer
        activate (str): Activation function type
        batch_size1 (int): Batch size for training
        random_seeds1 (int): Random seed for reproducibility

    Returns:
        MLP: Trained model
    """
    # ------------------- Data splitting -------------------
    train_num = round(INPUT.size(0) * bili)  # Number of training samples
    train_input = INPUT[:train_num]  # Training inputs
    train_output = OUTPUT[:train_num]  # Training targets
    test_input = INPUT[train_num:]  # Test inputs
    test_output = OUTPUT[train_num:]  # Test targets

    # Get input/output dimensions
    input_dimension = INPUT.size(1)
    output_dimension = 1  # Single-value regression task

    hidden_channels = hidden_channels1

    # ------------------- Model, optimizer, loss -------------------
    model = MLP(
        input_dimension=input_dimension,
        hidden_channels=hidden_channels,
        output_dimension=output_dimension,
        n_hidden_layer=n_layer,
        activate=activate,
        random_seeds=random_seeds1
    )

    # AdamW optimizer (Adam + decoupled weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr1,
        weight_decay=weight_decay1
    )

    # Mean Squared Error loss (standard for regression)
    criterion = nn.MSELoss()

    # Create dataset and dataloader for training
    train_dataset = TensorDataset(train_input, train_output)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size1,
        shuffle=True
    )

    # ------------------- Single epoch training function -------------------
    def train():
        model.train()  # Set model to training mode
        total_loss = 0.0
        num_samples = 0

        for batch_input, batch_output in train_loader:
            out = model(batch_input)  # Forward pass
            loss = criterion(out, batch_output)  # Compute loss

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            # Accumulate loss (weighted by actual batch size)
            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        # Return average loss for the epoch
        return total_loss / num_samples if num_samples > 0 else float('inf')

    # ------------------- Main training loop -------------------
    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)  # Tensor to store loss history
    k = 0

    for epoch in range(1, Epoch + 1):
        loss = train()  # Run one training epoch
        loss1[k] = loss
        k += 1

        # Early stopping condition: loss is sufficiently small
        if loss <= 0.0001:
            break

    # Return the trained model
    return model



