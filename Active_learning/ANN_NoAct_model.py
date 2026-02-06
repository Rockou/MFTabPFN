import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import TensorDataset, DataLoader

class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_channels, output_dimension, n_hidden_layer, random_seeds):
        super(MLP, self).__init__()
        torch.manual_seed(random_seeds)
        self.input_layer = nn.Linear(input_dimension, hidden_channels)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(n_hidden_layer - 1)])
        self.output_layer = nn.Linear(hidden_channels, output_dimension)
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

def SingleANN_NoAct_M(INPUT, OUTPUT, bili, n_layer, lr1, weight_decay1, Epoch1, hidden_channels1, batch_size1, random_seeds1):
    train_num = round(INPUT.size(0) * bili)
    train_input = INPUT[:train_num]
    train_output = OUTPUT[:train_num]
    test_input = INPUT[train_num:]
    test_output = OUTPUT[train_num:]

    input_dimension = INPUT.size(1)
    output_dimension = 1
    hidden_channels = hidden_channels1
    model = MLP(input_dimension=input_dimension, hidden_channels=hidden_channels, output_dimension=output_dimension, n_hidden_layer=n_layer, random_seeds=random_seeds1)  # 64

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr1, weight_decay=weight_decay1)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_input, train_output)
    train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)

    def train():
        model.train()
        total_loss = 0.0
        num_samples = 0
        for batch_input, batch_output in train_loader:
            out = model(batch_input)
            loss = criterion(out, batch_output)
            optimizer.zero_grad()
            loss.backward()
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
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if loss <= 0.0001:
            break

    return model


