import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import TensorDataset, DataLoader
import copy


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, random_seeds):
        super(SpectralConv1d, self).__init__()
        torch.manual_seed(random_seeds)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def complex_mul1d(self, input, weights):
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        actual_modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :actual_modes] = self.complex_mul1d(x_ft[:, :, :actual_modes], self.weights[:, :, :actual_modes])

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, random_seeds):
        super(FNO1d, self).__init__()
        torch.manual_seed(random_seeds)

        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(in_dim, width)

        # FNO Layers
        self.conv0 = SpectralConv1d(width, width, modes, random_seeds)
        self.w0 = nn.Conv1d(width, width, 1)

        self.conv1 = SpectralConv1d(width, width, modes, random_seeds)
        self.w1 = nn.Conv1d(width, width, 1)

        # Projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_dim)

        self.act = nn.GELU()

    def forward(self, x):
        # x input shape: [Batch, Input_Dim] -> [Batch, 1, Input_Dim]
        x = x.unsqueeze(1)
        # Lifting
        x = self.fc0(x)  # [Batch, 1, Width]
        # Permute for Conv1d: [Batch, Width, Length=1]
        x = x.permute(0, 2, 1)
        # Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)
        # Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)
        # Permute back: [Batch, Length=1, Width]
        x = x.permute(0, 2, 1)
        # Projection
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)  # [Batch, 1, Output_Dim]
        return x.squeeze(1)


def SingleFNO_M(INPUT, OUTPUT, N_low, N_high, modes, lr_low, lr_high, weight_decay_low, weight_decay_high, Epoch_low, Epoch_high, width, batch_size, random_seeds):
    train_num = round(INPUT.size(0))
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    # test_input  = INPUT[train_num:]
    # test_output = OUTPUT[train_num:]

    train_input_low = train_input[:N_low]
    train_input_high = train_input[N_low:]
    train_output_low = train_output[:N_low]
    train_output_high = train_output[N_low:]

    input_dimension = INPUT.size(1)
    output_dimension = 1
    model_low = FNO1d(in_dim=input_dimension, out_dim=output_dimension, modes=modes, width=width, random_seeds=random_seeds)
    optimizer_low = torch.optim.AdamW(model_low.parameters(), lr=lr_low, weight_decay=weight_decay_low)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_input_low, train_output_low)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model_low.train()
    for epoch in range(Epoch_low):
        total_loss = 0.0
        num_samples = 0
        for batch_input, batch_output in train_loader:
            out = model_low(batch_input)
            loss = criterion(out, batch_output)
            optimizer_low.zero_grad()
            loss.backward()
            optimizer_low.step()
            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        loss_low = total_loss / num_samples if num_samples > 0 else float('inf')
        # print(f'Epoch: {epoch:03d}, Loss: {loss_low:.6f}')

    pretrained_weights = copy.deepcopy(model_low.state_dict())
    model_high = FNO1d(in_dim=input_dimension, out_dim=output_dimension, modes=modes, width=width, random_seeds=random_seeds)
    model_high.load_state_dict(pretrained_weights)

    optimizer_high = torch.optim.AdamW(model_high.parameters(), lr=lr_high, weight_decay=weight_decay_high)

    train_dataset = TensorDataset(train_input_high, train_output_high)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model_high.train()
    for epoch in range(Epoch_high):
        total_loss = 0.0
        num_samples = 0
        for batch_input, batch_output in train_loader:
            out = model_high(batch_input)
            loss = criterion(out, batch_output)
            optimizer_high.zero_grad()
            loss.backward()
            optimizer_high.step()
            batch_size_actual = batch_input.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        loss_high = total_loss / num_samples if num_samples > 0 else float('inf')
        # print(f'Epoch: {epoch:03d}, Loss: {loss_high:.6f}')

    return model_high, model_low

