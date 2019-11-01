import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn.models import Node2Vec
from torch.optim import Adam
from torch.nn import NLLLoss

class Node2VecModel(torch.nn.Module):
    def __init__(self, *args, lr=0.01, **kwargs):
        super(Node2VecModel, self).__init__()
        self.model = Node2Vec(*args, **kwargs)
        
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def forward(self, data):
        return self.model(data)
        
    def fit(self, data, epochs=10):
        data_loader = DataLoader(torch.arange(data.num_nodes), batch_size=64, shuffle=True)
        self.train()
        history = []
        for epoch in range(epochs):
            running_loss = 0.0
            for subset in data_loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(data.edge_index, subset)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print("---> ({}/{}) Running loss: {}".format(epoch+1, epochs, running_loss / len(subset)))
            history.append(running_loss / len(subset))

        return history
