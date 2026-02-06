import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress deprecation warnings (common when using older PyTorch APIs or third-party libraries)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from torch.utils.data import TensorDataset, DataLoader


class CNNRegressor(nn.Module):
    """
    1D Convolutional regressor with residual connection and global average pooling.
    """

    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds):
        """
        Args:
            input_dimension   : length of each input sample (treated as 1D sequence)
            hidden_channels   : number of channels after first conv layer
            output_dimension  : Number of output units
            n_hidden_layer    : number of convolutional blocks
            activate          : final activation before linear head ('relu','tanh','sigmoid')
            random_seeds      : for reproducible weight initialization
        """
        super(CNNRegressor, self).__init__()
        torch.manual_seed(random_seeds)

        self.input_dim = input_dimension
        self.hidden_channels = hidden_channels
        self.output_dim = output_dimension

        # Stack of Conv1d → BatchNorm → ReLU blocks
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for i in range(n_hidden_layer):
            out_channels = hidden_channels if i == 0 else hidden_channels // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        # 1×1 convolution to project residual (input) to match final conv channels
        self.residual_fc = nn.Conv1d(1, hidden_channels // 2, kernel_size=1)

        # Global average pooling across sequence dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final linear layer producing the correction
        self.output_fc = nn.Linear(hidden_channels // 2, output_dimension)

        # Activation before final projection (configurable)
        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

        # Learnable scalar factor for initial TabPFN prediction
        self.afa = nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        """
        Forward pass:
            Input:  (batch_size, input_dimension)
            Output: (batch_size, output_dimension)
        """
        residual = x.unsqueeze(1)  # → (batch, 1, length)
        x = residual

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            # Residual connection applied only at the last convolutional block
            if i == len(self.conv_layers) - 1:
                residual = self.residual_fc(residual)
                if x.shape == residual.shape:
                    x = x + residual
                    x = F.relu(x)

        x = self.global_pool(x)  # → (batch, channels, 1)
        x = x.squeeze(-1)  # → (batch, channels)

        x = self.act_fn(x)
        output = self.output_fc(x)
        return output


class CNNRegressor_fixed(nn.Module):
    """
    Variant of CNNRegressor where the scalar 'afa' is fixed (not learnable).
    Used when you want to freeze the weighting factor of the initial TabPFN prediction.
    """

    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate, random_seeds, afa):
        super(CNNRegressor_fixed, self).__init__()
        torch.manual_seed(random_seeds)

        self.input_dim = input_dimension
        self.hidden_channels = hidden_channels
        self.output_dim = output_dimension

        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for i in range(n_hidden_layer):
            out_channels = hidden_channels if i == 0 else hidden_channels // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        self.residual_fc = nn.Conv1d(1, hidden_channels // 2, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(hidden_channels // 2, output_dimension)

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

        # Fixed (non-learnable) scalar multiplier
        self.afa = nn.Parameter(torch.tensor(afa, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        residual = x.unsqueeze(1)
        x = residual

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i == len(self.conv_layers) - 1:
                residual = self.residual_fc(residual)
                if x.shape == residual.shape:
                    x = x + residual
                    x = F.relu(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.act_fn(x)
        output = self.output_fc(x)
        return output


def RCNN_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr1, weight_decay1, Epoch1,
           hidden_channels1, activate, batch_size1, N_low, N_high, TabPFN_prediction_initial1,
           random_seeds1, afa1, afa_index):
    """
    RCNN training function with learnable / fixed scaling factor 'afa'.

    Selects subset of training data based on N_low / N_high.
    Trains CNN to produce corrections, then feeds corrected features into TabPFN again.
    Loss is computed between TabPFN(new features) and true residual (adjusted by afa).

    Args:
        task_type                   : task identifier passed to TabPFN
        INPUT / OUTPUT              : full dataset tensors
        input_TabPFN                : input dimension of TabPFN
        bili                        : train split ratio
        N_low, N_high               : controls which part of training data is used
        TabPFN_prediction_initial1  : initial TabPFN predictions
        afa1                        : initial / fixed value of scaling factor alpha
        afa_index                   : 0 → learn afa, 1 → fix afa
    """
    # Split into train / test (only train is used in this function)
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    test_input = INPUT[train_num:]
    test_output = OUTPUT[train_num:]

    # Further split training data into "low" and "high" parts
    train_input_low = train_input[:N_low]
    train_input_high = train_input[N_low:]
    train_output_low = train_output[:N_low]
    train_output_high = train_output[N_low:]

    input_dimension = INPUT.size(1)
    output_dimension = input_TabPFN
    hidden_channels = hidden_channels1

    # Choose model variant: learnable afa or fixed afa
    if afa_index == 0:
        model = CNNRegressor(
            input_dimension=input_dimension,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
            n_hidden_layer=n_layer,
            activate=activate,
            random_seeds=random_seeds1
        )
    elif afa_index == 1:
        model = CNNRegressor_fixed(
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

    # Select which subset to train on
    if N_high == 0:
        train_dataset = TensorDataset(train_input_low, train_output_low, TabPFN_prediction_initial)
    else:
        train_dataset = TensorDataset(train_input_high, train_output_high, TabPFN_prediction_initial)

    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    def train():
        model.train()
        total_loss = 0.0
        num_samples = 0

        save_path = Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt"

        for batch_input, batch_output, TabPFN_pred_low_high in train_loader:
            out = model(batch_input)

            # Re-run TabPFN on corrected features
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path,
                X_train=out,
                y_train=batch_output - model.afa * TabPFN_pred_low_high,
                X_test=out,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Target = true value - α × initial TabPFN prediction
            target = batch_output - model.afa * TabPFN_pred_low_high

            loss = criterion(TabPFN_prediction_tensor.to('cpu'), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        return total_loss / num_samples if num_samples > 0 else float('inf')

    # Training loop with early stopping
    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)
    k = 0

    for epoch in range(1, Epoch + 1):
        loss = train()
        loss1[k] = loss
        k += 1
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        # print(model.afa.detach().numpy())

        if loss <= 0.0001:
            break

    return model

