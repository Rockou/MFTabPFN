import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress deprecation warnings (common when using older parts of PyTorch or third-party libs)
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

        # Build stack of conv → BN → ReLU blocks
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

        # 1×1 conv used to project residual (original input) to match last conv channels
        self.residual_fc = nn.Conv1d(1, hidden_channels // 2, kernel_size=1)

        # Global average pooling → collapses sequence dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final linear layer producing the correction / refined prediction
        self.output_fc = nn.Linear(hidden_channels // 2, output_dimension)

        # Activation applied before the final linear layer (uncommon but user-configurable)
        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid

    def forward(self, x):
        """
        Forward pass:
            Input shape: (batch_size, input_dimension)
            Output shape: (batch_size, output_dimension)
        """
        # Add channel dimension → (batch, 1, length)
        residual = x.unsqueeze(1)
        x = residual

        # Pass through convolutional blocks
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

            # Add residual connection only at the last conv layer
            if i == len(self.conv_layers) - 1:
                residual = self.residual_fc(residual)
                if x.shape == residual.shape:
                    x = x + residual
                    x = F.relu(x)

        # Global average pooling over the length dimension
        x = self.global_pool(x)  # → (batch, channels, 1)
        x = x.squeeze(-1)  # → (batch, channels)

        # Optional activation before final projection
        x = self.act_fn(x)

        # Final linear prediction
        output = self.output_fc(x)
        return output


def RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr1, weight_decay1, Epoch1, hidden_channels1,
           activate, batch_size1, TabPFN_prediction_initial1, random_seeds1):
    """
    Residual CNN corrector that learns to refine TabPFN predictions.
    Trains on the residual (true - TabPFN_pred) using a RCNN.
    Args:
        task_type                   : passed to TabPFN_model_main
        INPUT                       : (N, input_dimension) feature matrix
        OUTPUT                      : (N, 1) target vector
        input_TabPFN                : TabPFN input dim
        bili                        : train / total ratio
        n_layer                     : number of layers
        lr1, weight_decay1          : AdamW hyperparameters
        Epoch1                      : maximum epochs
        hidden_channels1            : base number of conv filters
        activate                    : activation before final FC
        batch_size1                 : training batch size
        TabPFN_prediction_initial1  : initial TabPFN predictions on training set
        random_seeds1               : for model initialization reproducibility
    """
    # Split data — only training portion is used here
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]

    input_dimension = INPUT.size(1)
    output_dimension = input_TabPFN
    hidden_channels = hidden_channels1

    # Create RCNN
    model = CNNRegressor(
        input_dimension=input_dimension,
        hidden_channels=hidden_channels,
        output_dimension=output_dimension,
        n_hidden_layer=n_layer,
        activate=activate,
        random_seeds=random_seeds1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr1,
        weight_decay=weight_decay1,
    )

    criterion = nn.MSELoss()

    # Prepare residual targets: y - TabPFN(ŷ)
    TabPFN_prediction_initial = TabPFN_prediction_initial1
    Train_output_delt = train_output.detach().numpy() - TabPFN_prediction_initial
    train_output_delt = torch.tensor(Train_output_delt, dtype=torch.float)

    train_dataset = TensorDataset(train_input, train_output_delt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    def train():
        """One training epoch — calls TabPFN inside loop."""
        model.train()
        total_loss = 0.0
        num_samples = 0

        # Path where fine-tuned TabPFN checkpoint would be saved/loaded
        save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")

        for batch_input, batch_output_delt in train_loader:
            out = model(batch_input)

            # Re-run TabPFN on the current batch correction
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

            # Loss = MSE between TabPFN(new_features) and true residual
            loss = criterion(TabPFN_prediction_tensor.to('cpu'), batch_output_delt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        return total_loss / num_samples if num_samples > 0 else float('inf')

    # Training loop with simple early stopping
    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)
    k = 0

    for epoch in range(1, Epoch + 1):
        loss = train()
        loss1[k] = loss
        k = k + 1
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if loss <= 0.0001:
            break

    return model

