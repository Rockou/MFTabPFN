import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress deprecation warnings (common with older PyTorch APIs or third-party libraries)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from torch.utils.data import TensorDataset, DataLoader


class MLP(torch.nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) used as a feature transformer / corrector
    for refining TabPFN predictions in a residual-learning style.
    """
    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds):
        """
        Args:
            input_dimension   : number of input features per sample
            hidden_channels   : number of neurons in each hidden layer
            output_dimension  : dimension of output
            n_hidden_layer    : number of layers
            activate          : activation function type ('relu', 'tanh', 'sigmoid')
            random_seeds      : seed for reproducible weight initialization
        """
        super(MLP, self).__init__()

        # Fix random seed for reproducible initialization
        torch.manual_seed(random_seeds)

        # Input → first hidden layer
        self.input_layer = nn.Linear(input_dimension, hidden_channels)

        # Intermediate hidden layers (if more than one)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_hidden_layer - 1)
        ])

        # Last hidden layer → output
        self.output_layer = nn.Linear(hidden_channels, output_dimension)

        # Select activation function (defaults to tanh)
        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

    def forward(self, x):
        """
        Forward pass:
            Input shape:  (batch_size, input_dimension)
            Output shape: (batch_size, output_dimension)
        """
        # First layer + activation
        x = self.act_fn(self.input_layer(x))

        # All intermediate hidden layers + activation
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x))

        # Output layer
        output = self.output_layer(x)
        return output


def MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr1, weight_decay1, Epoch1,
          hidden_channels1, activate, batch_size1, TabPFN_prediction_initial1, random_seeds1):
    """
    Trains an MLP to act as a residual corrector for TabPFN predictions.

    The MLP transforms input features → new features → feed into TabPFN again.
    Loss is computed between TabPFN(new_features) and the true residual (y - initial TabPFN pred).

    Args:
        task_type                   : task type string passed to TabPFN
        INPUT                       : (N, d) feature matrix
        OUTPUT                      : (N, 1) target vector
        input_TabPFN                : output dimension of TabPFN (typically 1)
        bili                        : train / total data ratio
        n_layer                     : number of layers in MLP
        lr1, weight_decay1          : AdamW hyperparameters
        Epoch1                      : max training epochs
        hidden_channels1            : hidden layer width
        activate                    : activation function type
        batch_size1                 : training batch size
        TabPFN_prediction_initial1  : initial TabPFN predictions on training set
        random_seeds1               : random seed for model initialization
    """
    # ------------------- Data preparation -------------------
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]

    input_dimension = INPUT.size(1)
    output_dimension = input_TabPFN
    hidden_channels = hidden_channels1

    # Initialize MLP corrector
    model = MLP(
        input_dimension=input_dimension,
        hidden_channels=hidden_channels,
        output_dimension=output_dimension,
        n_hidden_layer=n_layer,
        activate=activate,
        random_seeds=random_seeds1
    )

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr1,
        weight_decay=weight_decay1,
    )
    criterion = nn.MSELoss()

    # Prepare residual targets: y - initial TabPFN prediction
    TabPFN_prediction_initial = TabPFN_prediction_initial1
    Train_output_delt = train_output.detach().numpy() - TabPFN_prediction_initial
    train_output_delt = torch.tensor(Train_output_delt, dtype=torch.float)

    train_dataset = TensorDataset(train_input, train_output_delt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    # ------------------- Training step (one epoch) -------------------
    def train():
        model.train()
        total_loss = 0.0
        num_samples = 0

        # Checkpoint path for (potentially) fine-tuned TabPFN
        save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")

        for batch_input, batch_output_delt in train_loader:
            # MLP transforms original features
            out = model(batch_input)

            # Re-run TabPFN on the transformed features
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=out,
                y_train=batch_output_delt,
                X_test=out,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Loss: how well TabPFN(new features) matches the true residual
            loss = criterion(TabPFN_prediction_tensor.to('cpu'), batch_output_delt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        return total_loss / num_samples if num_samples > 0 else float('inf')

    # ------------------- Training loop -------------------
    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)
    k = 0

    for epoch in range(1, Epoch + 1):
        loss = train()
        loss1[k] = loss
        k += 1
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # Early stopping condition
        if loss <= 0.0001:
            break

    return model


