import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from anomuly import ANOMULY

device = torch.device("cpu")

def anomaly_loss(logits, labels):
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def load_data(dataset):
    data = Planetoid(root="./", name=dataset)
    data.x = data.x.float()
    data.edge_index = data.edge_index.to(torch.long)
    return data


def train(model, data, optimizer, epoch):
    model.train()
    # data = data.to(device)
    logits, _ = model(data)
    loss = anomaly_loss(logits, data.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    # data = data.to(device)
    logits, _ = model(data)
    loss = anomaly_loss(logits, data.y)
    return loss.item()


def main():
    dataset = 'Cora'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(dataset)

    x = torch.randn(8, 1433, 16)
    # i have 1433 node, generate 1000 edges index for me
    edge_index = torch.randint(0, 1433, (8, 2, 1000))

    data.x = x
    data.edge_index = edge_index

    n_views = x.shape[0]
    n_layers = 2

    hidden_dims = [128 for _ in range(n_views)]

    gru_hidden_dims = [128 for _ in range(n_views)]
    attns_in_channels = [128 for _ in range(n_layers)]
    attns_out_channels = [128 for _ in range(n_layers)]


    model = ANOMULY(
        n_views=n_views,
        num_features=x.shape[2],
        hidden_dims=hidden_dims,
        gru_hidden_dims=gru_hidden_dims,
        attns_in_channels=attns_in_channels,
        attns_out_channels=attns_out_channels,
        n_layers=n_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    for epoch in range(epochs):
        train_loss = train(model, data, optimizer, epoch)
        test_loss = test(model, data)
        print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}")


if __name__ == "__main__":
    main()
