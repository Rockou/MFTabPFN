import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tabpfn import TabPFNRegressor
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from torch.utils.data import TensorDataset, DataLoader

class CNNRegressor(nn.Module):
    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, activate='none'):
        super(CNNRegressor, self).__init__()
        torch.manual_seed(1234)
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
    def forward(self, x):
        residual = x.unsqueeze(1)  # (batch_size, 1, input_dim)
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

def RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr1, weight_decay1, Epoch1, hidden_channels1, activate, batch_size1):
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    test_input = INPUT[train_num:]
    test_output = OUTPUT[train_num:]

    input_dimension = INPUT.size(1)
    output_dimension = input_TabPFN
    hidden_channels = hidden_channels1
    model = CNNRegressor(
        input_dimension=input_dimension,
        hidden_channels=hidden_channels,
        output_dimension=output_dimension,
        n_hidden_layer=n_layer,
        activate=activate
    )
    # print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr1,
        weight_decay=weight_decay1,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    warmup_epochs = 8
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.MSELoss()

    if input_dimension > 500:
        reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
    else:
        reg = TabPFNRegressor(n_estimators=8, random_state=42)
    reg.fit(train_input.detach().numpy(), train_output.detach().numpy().ravel())
    TabPFN_prediction_initial = reg.predict(train_input.detach().numpy()).reshape(-1, 1)
    Train_output_delt = train_output.detach().numpy() - TabPFN_prediction_initial
    train_output_delt = torch.tensor(Train_output_delt, dtype=torch.float)
    train_dataset = TensorDataset(train_input, train_output_delt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    def train():
        model.train()
        total_loss = 0.0
        num_samples = 0

        save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn_model_{task_type}.ckpt")
        for batch_input, batch_output_delt in train_loader:
            out = model(batch_input)
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=out,
                y_train=batch_output_delt,
                X_test=out,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                # device="cuda" if torch.cuda.is_available() else "cpu",
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

            loss = criterion(TabPFN_prediction_tensor.to('cpu'), batch_output_delt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        return total_loss / num_samples if num_samples > 0 else float('inf')

    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)
    k = 0
    for epoch in range(1, Epoch + 1):
        loss = train()
        loss1[k] = loss
        k = k + 1
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        scheduler.step(loss)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        if loss <= 0.0001:
            break

    return model
