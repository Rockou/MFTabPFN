import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress deprecation warnings (common with older PyTorch APIs or external libraries)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from torch.utils.data import TensorDataset, DataLoader


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) that learns a feature transformation / correction
    to improve TabPFN predictions in a residual style.

    Contains a learnable scalar parameter `afa` (alpha) that scales the initial
    TabPFN prediction when computing the target residual.
    """

    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds):
        """
        Args:
            input_dimension   : input feature dimension
            hidden_channels   : number of neurons in each hidden layer
            output_dimension  : output dimension
            n_hidden_layer    : number of layers
            activate          : activation function type ('relu', 'tanh', 'sigmoid')
            random_seeds      : seed for reproducible weight initialization
        """
        super(MLP, self).__init__()
        torch.manual_seed(random_seeds)

        self.input_layer = nn.Linear(input_dimension, hidden_channels)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_hidden_layer - 1)
        ])
        self.output_layer = nn.Linear(hidden_channels, output_dimension)

        # Select activation function (defaults to tanh)
        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

        # Learnable scalar weighting factor for initial TabPFN prediction
        self.afa = nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        """
        Forward pass:
            Input:  (batch_size, input_dimension)
            Output: (batch_size, output_dimension)
        """
        x = self.act_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x))
        output = self.output_layer(x)
        return output


class MLP_fixed(torch.nn.Module):
    """
    Variant of MLP where the scalar `afa` (alpha) is fixed and not trainable.
    Used when you want to freeze the weighting of the initial TabPFN prediction.
    """

    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds, afa):
        super(MLP_fixed, self).__init__()
        torch.manual_seed(random_seeds)

        self.input_layer = nn.Linear(input_dimension, hidden_channels)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_hidden_layer - 1)
        ])
        self.output_layer = nn.Linear(hidden_channels, output_dimension)

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

        # Fixed (non-learnable) scalar multiplier
        self.afa = nn.Parameter(torch.tensor(afa, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        x = self.act_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x))
        output = self.output_layer(x)
        return output


def MLP_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr1, weight_decay1, Epoch1,
          hidden_channels1, activate, batch_size1, N_low, N_high, TabPFN_prediction_initial1,
          random_seeds1, afa1, afa_index):
    """
    Trains an MLP-based residual corrector with optional learnable/fixed scaling factor `afa`.

    Splits training data into "low" and "high" parts (controlled by N_low / N_high).
    Trains MLP to generate new features → feeds them into TabPFN → minimizes difference
    between TabPFN output and adjusted residual (y - α × initial TabPFN pred).

    Args:
        task_type                   : task identifier passed to TabPFN
        INPUT / OUTPUT              : full feature and target tensors
        input_TabPFN                : TabPFN input dimension
        bili                        : training set proportion
        N_low / N_high              : controls which subset of training data is used
        TabPFN_prediction_initial1  : initial TabPFN predictions
        afa1                        : initial or fixed value of scaling factor α
        afa_index                   : 0 = learn afa, 1 = fix afa
    """
    # ------------------- Data splitting -------------------
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    test_input = INPUT[train_num:]
    test_output = OUTPUT[train_num:]

    # Subset selection: low vs high part of training data
    train_input_low = train_input[:N_low]
    train_input_high = train_input[N_low:]
    train_output_low = train_output[:N_low]
    train_output_high = train_output[N_low:]

    input_dimension = INPUT.size(1)
    output_dimension = input_TabPFN
    hidden_channels = hidden_channels1

    # Choose model variant: learnable or fixed afa
    if afa_index == 0:
        model = MLP(
            input_dimension=input_dimension,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
            n_hidden_layer=n_layer,
            activate=activate,
            random_seeds=random_seeds1
        )
    elif afa_index == 1:
        model = MLP_fixed(
            input_dimension=input_dimension,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
            n_hidden_layer=n_layer,
            activate=activate,
            random_seeds=random_seeds1,
            afa=afa1
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr1,
        weight_decay=weight_decay1,
    )
    criterion = nn.MSELoss()

    # Convert initial TabPFN predictions to fixed tensor
    TabPFN_prediction_initial = torch.tensor(
        TabPFN_prediction_initial1,
        dtype=torch.float,
        requires_grad=False
    )

    # Select training subset
    if N_high == 0:
        train_dataset = TensorDataset(train_input_low, train_output_low, TabPFN_prediction_initial)
    else:
        train_dataset = TensorDataset(train_input_high, train_output_high, TabPFN_prediction_initial)

    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    # ------------------- Single epoch training function -------------------
    def train():
        model.train()
        total_loss = 0.0
        num_samples = 0

        save_path = Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt"

        for batch_input, batch_output, TabPFN_pred_batch in train_loader:
            out = model(batch_input)

            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path,
                X_train=out,
                y_train=batch_output - model.afa * TabPFN_pred_batch,
                X_test=out,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
                # device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

            # Target residual: y - α × initial TabPFN prediction
            target = batch_output - model.afa * TabPFN_pred_batch

            loss = criterion(TabPFN_prediction_tensor.to('cpu'), target)

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
        # print(model.afa.detach().numpy())  # show current value of alpha

        if loss <= 0.0001:
            break

    return model


