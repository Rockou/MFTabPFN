import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import TensorDataset, DataLoader


class RBFNN(nn.Module):

    def __init__(self, input_dim, num_centers, output_dim, random_seeds):
        super(RBFNN, self).__init__()
        torch.manual_seed(random_seeds)

        self.num_centers = num_centers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))

        self.sigma = nn.Parameter(torch.ones(num_centers))

        self.linear = nn.Linear(num_centers, output_dim, bias=True)

    def forward(self, x):
        # x: [batch_size, input_dim]
        # centers: [num_centers, input_dim]

        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # [batch, num_centers, input_dim]
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [batch, num_centers]

        # exp(-||x-c||² / (2*sigma²))
        rbf = torch.exp(-dist_sq / (2 * self.sigma ** 2 + 1e-8))  # [batch, num_centers]

        out = self.linear(rbf)  # [batch, output_dim]

        return out


def SingleRBFNN_M(INPUT, OUTPUT, bili, num_centers, lr, weight_decay, epochs, batch_size, random_seeds):

    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    test_input = INPUT[train_num:]
    test_output = OUTPUT[train_num:]

    input_dim = INPUT.size(1)
    output_dim = OUTPUT.size(1) if OUTPUT.dim() > 1 else 1

    model = RBFNN(
        input_dim=input_dim,
        num_centers=num_centers,
        output_dim=output_dim,
        random_seeds=random_seeds
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_input, train_output)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train_one_epoch():
        model.train()
        total_loss = 0.0
        num_samples = 0

        for batch_x, batch_y in train_loader:
            out = model(batch_x)
            loss = criterion(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_x.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        return total_loss / num_samples if num_samples > 0 else float('inf')

    loss_history = torch.zeros(epochs)
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch()
        loss_history[epoch - 1] = loss
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if loss < 0.0001:
            # print("Early stopping triggered")
            break

    return model


