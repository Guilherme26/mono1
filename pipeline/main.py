import pandas as pd
import numpy as np
import torch
import preprocessing
import utils

from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from collections import defaultdict


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    profiles = pd.read_csv("../data/profiles.csv", usecols=["profile_username", "profile_followed_by", "profile_follow", 
                                                            "medias_nb", "comments_nb", "comments_commenters_nb", 
                                                            "comments_self_nb", "category_1"])
    comments = pd.read_csv("../data/comments.csv", usecols=["media_short_code", "media_author", "commenter"])

    profiles = preprocessing.categorical_to_numerical(profiles, col="category_1")
    comments = comments.drop_duplicates()
    comments = preprocessing.filter_by_relevance(comments, profiles, minimum_freq=50)

    known_users = profiles.profile_username.unique().tolist()
    followers = comments.commenter.unique().tolist()
    all_users = set(known_users + followers)

    names = profiles.profile_username.values
    data = profiles[["profile_followed_by", "profile_follow", "medias_nb", 
                    "comments_nb", "comments_commenters_nb", "comments_self_nb"]].values
    name_to_record = {name: record for name, record in zip(names, data)}

    input_dim, output_dim = data.shape[1], len(profiles.category_1.unique()) + 1
    user_to_label = {user: category for user, category in profiles[["profile_username", "category_1"]].values}

    skf = StratifiedKFold(n_splits=5)

    # for n_hidden_units in [64, 128, 256]:
    print("Fez of preprocessamento")
    n_hidden_units = 64
    for train_idx, test_idx in skf.split(profiles.profile_username.values, profiles.category_1.values):
        train_authors, test_authors = utils.get_authors(profiles, all_users, train_idx, test_idx)

        print("Pegou autores")

        username_to_index = utils.get_users_indices(train_authors)
        print("Pegou indices")
        train_interactions = utils.get_interactions(comments[(comments.media_author.isin(train_authors)) 
                                                        & (comments.commenter.isin(train_authors))], username_to_index)
        print("Pegou interações")
        x_train, y_train = utils.get_x(train_authors, name_to_record, input_dim=input_dim), utils.get_y(user_to_label, train_authors)
        print("Pegou x e y")
        assert len(x_train)==len(y_train), "Input and Output tensor do not have the same dimensions"


        edge_index = utils.get_edge_index(train_interactions)
        print("Pegou os indices de arestas")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = Data(x=x_train, y=y_train, edge_index=edge_index).to(device)
        print("Criou Data")

        models = utils.get_models(data.num_nodes, input_dim, output_dim, n_hidden_units, device=device)
        print("Criou modelos")

        utils.train(data, models)

        # username_to_index, users_indices = utils.get_users_indices(test_authors)
        # test_interactions = utils.get_interactions(comments[(comments.media_author.isin(test_authors)) 
                                                        # & (comments.commenter.isin(test_authors))], username_to_index)
        # x_train, y_train = utils.get_x(user_indices, index_to_record), utils.get_y(user_to_label, test_authors)
        # edge_index = utils.get_edge_index(test_interactions)

if __name__ == "__main__":
    main()