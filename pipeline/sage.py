import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch.optim import Adam
from torch.nn import NLLLoss


class GraphSAGE(torch.nn.Module):
    def __init__(self, n_features, n_hidden_units, n_classes, lr=0.01, n_hidden_layers=1):
        super(GraphSAGE, self).__init__()
        self.convs = [SAGEConv(n_features, n_hidden_units)] + [SAGEConv(n_hidden_units, n_hidden_units) for _ in range(n_hidden_layers-1)]
        self.convs = torch.nn.Sequential(*self.convs)
        self.output = SAGEConv(n_hidden_units, n_classes)
        
        self.loss = NLLLoss()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=5e-4)
    
    def forward(self, x, edge_index, apply_activation=True):
        for layer in self.convs:
            x = F.relu(layer(x, edge_index))
        return F.log_softmax(self.output(x, edge_index), dim=1) if apply_activation else x
    
    def fit(self, data, epochs=10):
        self.train()
        history = []
        for epoch in range(epochs):
            outputs = self.forward(data.x, data.edge_index)
            loss = self.loss(outputs[data.train_mask], data.y[data.train_mask])
            loss.backward()
            
            self.optimizer.step()
            print("---> ({}/{}) Running loss: {}".format(epoch+1, epochs, loss.item()))
            history.append(loss.item())

        return history