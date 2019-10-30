import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from torch.nn import NLLLoss


class GCNModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden_units, n_classes, **kwargs):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(n_features, n_hidden_units, cached=True)
        self.conv2 = GCNConv(n_hidden_units, n_classes, cached=True)
        
        self.loss = NLLLoss()
        self.optimizer = Adam(self.parameters(), lr=0.01, weight_decay=5e-4)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    def fit(self, data, epochs=10):
        self.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()

            outputs = self.forward(data.x, data.edge_index)
            loss = self.loss(outputs, data.y)
            loss.backward()

            self.optimizer.step()
            print("GCN running loss is: {}".format(loss.item()))