import pandas as pd
import numpy as np
import torch
import preprocessing
import utils
import argparse

from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Available Parameters:")
    parser.add_argument("--n_hidden_units", default=64, type=int)
    parser.add_argument("--n_hidden_layers", default=1, type=int)
    parser.add_argument("--train_epochs", default=100, type=int)
    parser.add_argument("--write_output", default=True, type=bool)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    profiles = pd.read_csv("../data/new_profiles_200t.csv")
    comments = pd.read_csv("../data/new_comments_200t.csv")

    comments = comments.drop_duplicates()
    profiles = preprocessing.categorical_to_numerical(profiles, col="category_1")
    all_users = set(profiles.profile_username.values)

    data = preprocessing.scale(profiles.drop(columns=["category_1", "profile_username"]).values)
    name_to_record = {name: record for name, record in zip(all_users, data)}

    input_dim, output_dim = data.shape[1], len(profiles.category_1.unique()) + 1
    user_to_label = {user: category for user, category in profiles[["profile_username", "category_1"]].values}

    K = 5
    skf = StratifiedKFold(n_splits=K)
    models_metrics, models_histories = defaultdict(dict), defaultdict(list)
    for kth_fold, (train_idx, test_idx) in enumerate(skf.split(profiles.profile_username.values, profiles.category_1.values), start=1):
        print("Starting {}th Fold".format(kth_fold))

        authors = profiles.profile_username.values
        username_to_index = utils.get_users_indices(authors)
        interactions = utils.get_interactions(comments, username_to_index)
        edge_index = utils.get_edge_index(interactions)
        
        x = utils.get_x(authors, name_to_record, input_dim=input_dim)
        y = utils.get_y(user_to_label, authors)

        train_mask = [True if i in train_idx else False for i in range(len(x))]
        test_mask = [True if i in test_idx else False for i in range(len(x))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask).to(device)

        assert len(x)==len(y), "Train Input and Output tensor do not have the same dimensions"

        models = utils.get_models(data.num_nodes, input_dim, output_dim, args.n_hidden_units, args.n_hidden_layers, device=device, lr=0.005)
        histories = utils.train(data, models, epochs=args.train_epochs)
        models_histories = utils.update_histories(models_histories, histories)

        current_metrics = utils.test(data, models)
        utils.update_metrics_dict(models_metrics, current_metrics)

        print('\n')
        
    models_histories = {model: list(history/K) for model, history in models_histories.items()} # Get mean traces
    models_metrics = utils.calculate_statistics(models_metrics)

    if args.write_output:
        utils.write_json("../data/results/models_metrics_{}e_{}l_{}u.json".format(args.train_epochs, args.n_hidden_layers, args.n_hidden_units), models_metrics)
        utils.write_json("../data/results/models_histories_{}e_{}l_{}u.json".format(args.train_epochs, args.n_hidden_layers, args.n_hidden_units), models_histories)

if __name__ == "__main__":
    main()