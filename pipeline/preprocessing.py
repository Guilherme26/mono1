import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def categorical_to_numerical(df, col):
    categories = df[col].value_counts().index
    category_to_index = {category: index for index, category in enumerate(categories)}
    df[col] = df[col].map(lambda key: category_to_index[key])

    return df


def filter_by_relevance(comments, profiles, minimum_freq=30):
    known_users = profiles.profile_username.unique().tolist()
    relevant_users = []

    action_authors = pd.concat([comments.commenter.value_counts(), comments.media_author.value_counts()])
    action_authors = action_authors.groupby(action_authors.index).agg("sum")
    for user, frequency in action_authors.items():
        if (frequency > minimum_freq) or (user in known_users):
            relevant_users.append(user)
            
    return comments[comments.commenter.isin(relevant_users)]


def scale(data):
    return MinMaxScaler().fit_transform(data)