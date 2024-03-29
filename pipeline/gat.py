import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch.nn import NLLLoss


class GATModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden_units, n_classes, lr=0.01, n_hidden_layers=1):
        super(GATModel, self).__init__()
        self.convs = [GATConv(n_features, n_hidden_units, heads=8)] + [GATConv(8*n_hidden_units, n_hidden_units, heads=8) for _ in range(n_hidden_layers-1)]
        self.convs = torch.nn.Sequential(*self.convs)
        self.output = GATConv(8*n_hidden_units, n_classes, heads=8)

        self.loss = NLLLoss()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=5e-4)

    def forward(self, x, edge_index, apply_activation=True):
        for layer in self.convs:
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(layer(x, edge_index))
        return F.log_softmax(x, dim=1) if apply_activation else x
    
    def fit(self, data, epochs=10):
        self.train()
        history = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            outputs = self.forward(data.x, data.edge_index)
            loss = self.loss(outputs[data.train_mask], data.y[data.train_mask])
            loss.backward()
            
            self.optimizer.step()
            print("---> ({}/{}) Running loss: {}".format(epoch+1, epochs, loss.item()))
            history.append(loss.item())

        return history