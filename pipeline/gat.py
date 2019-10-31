import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch.nn import NLLLoss


class GATModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden_units, n_classes, lr=0.01):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(n_features, n_classes, heads=8, dropout=0.6)
        
        self.loss = NLLLoss()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=5e-4)

    def forward(self, x, edge_index, apply_activation=True):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)        
        return F.log_softmax(x, dim=1) if apply_activation else x
    
    def fit(self, data, epochs=10):
        self.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            outputs = self.forward(data.x, data.edge_index)
            loss = self.loss(outputs, data.y)
            loss.backward()
            
            self.optimizer.step()
            print("GAT running loss is: {}".format(loss.item()))