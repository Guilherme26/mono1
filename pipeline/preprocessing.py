import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def categorical_to_numerical(df, col):
    categories = df[col].value_counts().index
    category_to_index = {category: index for index, category in enumerate(categories)}
    df[col] = df[col].map(lambda key: category_to_index[key])

    return df


def filter_by_relevance(comments, profiles, minimum_freq=30):
    users = profiles.profile_username.unique().tolist()        
    return comments[(comments.commenter.isin(users)) | (comments.media_author.isin(users))]


def scale(data):
    return MinMaxScaler().fit_transform(data)