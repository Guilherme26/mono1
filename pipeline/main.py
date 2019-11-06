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

    profiles = pd.read_csv("../data/new_profiles.csv")
    comments = pd.read_csv("../data/comments.csv", usecols=["media_short_code", "media_author", "commenter"])

    profiles = preprocessing.categorical_to_numerical(profiles, col="category_1")
    comments = comments.drop_duplicates()
    comments = preprocessing.filter_by_relevance(comments, profiles)


    all_users = profiles.profile_username.values
    data = preprocessing.scale(profiles.drop(columns=["category_1", "profile_username"]).values)
    name_to_record = {name: record for name, record in zip(all_users, data)}

    input_dim, output_dim = data.shape[1], len(profiles.category_1.unique()) + 1
    user_to_label = {user: category for user, category in profiles[["profile_username", "category_1"]].values}

    K = 5
    skf = StratifiedKFold(n_splits=K)
    models_metrics, models_histories = defaultdict(dict), defaultdict(list)
    tracked_profiles = profiles[profiles.is_tracked == 1]
    for kth_fold, (train_idx, test_idx) in enumerate(skf.split(tracked_profiles.profile_username.values, tracked_profiles.category_1.values), start=1):
        print("Starting {}th Fold".format(kth_fold))

        train_authors, test_authors = utils.get_authors(profiles, all_users, train_idx, test_idx)
        username_to_index = utils.get_users_indices(train_authors)
        train_interactions = utils.get_interactions(comments[comments.media_author.isin(train_authors)], username_to_index)
        x_train, y_train = utils.get_x(train_authors, name_to_record, input_dim=input_dim), utils.get_y(user_to_label, train_authors)
        assert len(x_train)==len(y_train), "Train Input and Output tensor do not have the same dimensions"

        edge_index = utils.get_edge_index(train_interactions)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = Data(x=x_train, y=y_train, edge_index=edge_index).to(device)

        models = utils.get_models(data.num_nodes, input_dim, output_dim, args.n_hidden_units, args.n_hidden_layers, device=device, lr=0.005)
        histories = utils.train(data, models, epochs=args.train_epochs)
        models_histories = utils.update_histories(models_histories, histories)

        username_to_index = utils.get_users_indices(test_authors)
        test_interactions = utils.get_interactions(comments[comments.media_author.isin(test_authors)], username_to_index)
        x_test, y_test = utils.get_x(test_authors, name_to_record, input_dim=input_dim), utils.get_y(user_to_label, test_authors)
        assert len(x_test)==len(y_test), "Test Input and Output tensor do not have the same dimensions"

        edge_index = utils.get_edge_index(test_interactions)
        data = Data(x=x_test, y=y_test, edge_index=edge_index).to(device)
        current_metrics = utils.test(data, models)
        utils.update_metrics_dict(models_metrics, current_metrics)

        print('\n')
        
    models_histories = {model: list(history/K) for model, history in models_histories.items()} # Get mean traces
    models_metrics = utils.calculate_statistics(models_metrics)

    if args.write_output:
        utils.write_json("results/models_metrics_{}e_{}l_{}u.json".format(args.train_epochs, args.n_hidden_layers, args.n_hidden_units), models_metrics)
        utils.write_json("results/models_histories_{}e_{}l_{}u.json".format(args.train_epochs, args.n_hidden_layers, args.n_hidden_units), models_histories)

if __name__ == "__main__":
    main()