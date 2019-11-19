import torch
import json
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from n2v import Node2VecModel
from gcn import GCNModel
from gat import GATModel
from sage import GraphSAGE


def get_interactions(df, username_to_index):
    return [(username_to_index[author], username_to_index[commenter]) 
                for author, commenter in df[['media_author', 'commenter']].drop_duplicates().values]


def get_authors(df, train_idx, test_idx):
    test_users = df.iloc[test_idx].profile_username.values
    train_users = df.iloc[train_idx].profile_username.values
    return train_users, test_users


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


def get_models(n_nodes, input_dim, output_dim, n_hidden_units, n_hidden_layers, device='cpu', lr=0.01):
    models = [GCNModel(input_dim, n_hidden_units, output_dim, lr=lr, n_hidden_layers=n_hidden_layers),
              GATModel(input_dim, n_hidden_units, output_dim, lr=lr, n_hidden_layers=n_hidden_layers), 
              GraphSAGE(input_dim, n_hidden_units, output_dim, lr=lr, n_hidden_layers=n_hidden_layers)]
    
    return [model.to(device) for model in models]


def get_users_indices(authors):
    return {name: index for index, name in enumerate(authors)}


def train(data, models, epochs=10):
    train_traces = dict()
    for model in models:
        print("-> Beggining {}'s Training Process".format(model.__class__.__name__))
        train_traces[model.__class__.__name__] = model.fit(data, epochs=epochs)
        print('!=============================================================!')

    return train_traces
        

def test(data, models):
    metrics_per_model = {}
    for model in models:
        model.eval()
        y_preds = model.forward(data.x, data.edge_index)
        y_preds = torch.argmax(y_preds[data.test_mask], dim=1).detach().numpy()
        y_true = data.y[data.test_mask].detach().numpy()
        
        # print(classification_report(y_true, y_preds))
        print(f1_score(y_true, y_preds, average="macro"))
        
        metrics_per_model[model.__class__.__name__] = {"Accuracy": float(accuracy_score(y_true, y_preds)), 
                                                       "F1 Macro": float(f1_score(y_true, y_preds, average="macro")),
                                                       "F1 Micro": float(f1_score(y_true, y_preds, average="micro"))}

    return metrics_per_model


def update_metrics_dict(models_metrics, new_execution_dict):
    for model in new_execution_dict.keys():
        for metric in new_execution_dict[model].keys():
            models_metrics[model][metric] = models_metrics[model].get(metric, []) + [new_execution_dict[model][metric]]


def update_histories(models_histories, new_histories):
    for model, _ in new_histories.items():
        if list(models_histories[model]):
            models_histories[model] += np.array(new_histories[model])
        else:
            models_histories[model] = np.array(new_histories[model])
    
    return models_histories


def calculate_statistics(models_metrics):
    return {model: {metric: (np.mean(values), np.std(values)) for metric, values in metrics.items()} 
                for model, metrics in models_metrics.items()}


def write_json(file_name, dictionary):
    with open(file_name, 'w') as fp:
        json.dump(dictionary, fp)
