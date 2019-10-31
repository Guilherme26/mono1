import torch
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.metrics import f1_score, accuracy_score
from n2v import Node2VecModel
from gcn import GCNModel
from gat import GATModel
from sage import GraphSAGE


def get_interactions(df, username_to_index):
    return [(username_to_index[author], username_to_index[commenter]) 
                for author, commenter in df[['media_author', 'commenter']].drop_duplicates().values]


def get_authors(df, all_users, train_idx, test_idx):
    train_users = list(all_users - set(df.iloc[test_idx].profile_username.values))
    return train_users, df.iloc[test_idx].profile_username.values


def get_edge_index(interactions):
    graph = nx.Graph()
    graph.add_edges_from(interactions)
    
    return torch.tensor(nx.to_pandas_edgelist(graph).values.T, dtype=torch.long)

def get_x(authors, name_to_record, input_dim=6):
    x = [name_to_record.get(name, np.ones(input_dim)) for name in authors]
    return torch.tensor(x, dtype=torch.float)


def get_y(user_to_label, users):
    y = [user_to_label.get(user, 4) for user in users]
    return torch.tensor(y, dtype=torch.long)


def get_models(n_nodes, input_dim, output_dim, n_hidden_units, device='cpu', lr=0.01):
    models = [#Node2VecModel(n_nodes, embedding_dim=n_hidden_units, walk_length=20, context_size=10, walks_per_node=10, lr=lr), 
              GCNModel(input_dim, n_hidden_units, output_dim, lr=lr),
              GATModel(input_dim, n_hidden_units, output_dim, lr=lr), 
              GraphSAGE(input_dim, n_hidden_units, output_dim, lr=lr)]
    
    return [model.to(device) for model in models]


def get_users_indices(authors):
    return {name: index for index, name in enumerate(authors)}


def train(data, models, epochs=10):
    for model in models:
        model.fit(data, epochs=epochs)
        print("Treinou o Modelo {}".format(model))
        

def test(data, models):
    metrics_per_model = {}
    for model in models:
        model.eval()
        y_pred = torch.argmax(model.forward(data.x, data.edge_index), dim=1).detach().numpy()
        y_true = data.y.detach().numpy()
        metrics_per_model[model.__class__.__name__] = {"Accuracy": accuracy_score(y_true, y_pred), 
                                                       "F1 Macro": f1_score(y_true, y_pred, average="macro"),
                                                       "F1 Micro": f1_score(y_true, y_pred, average="micro")}

    return metrics_per_model