import pandas as pd
import numpy as np
import inspect
from sklearn.feature_selection import SelectPercentile, RFE
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as RFR


def eliminate_features(dataset, estimator, **kwargs):
    percentile = kwargs['percentile']
    num_of_features = kwargs['num_of_features']

    if np.floor((percentile/100)*(len(list(dataset.columns))-1)) <   num_of_features:
        raise ValueError("The number of features left after the "
                         "firsts stage of elimination is lesser than the final number of features! "
                         "Please spesify either a greater percentile or lesser number of final features!")

    # Preprocessing
    X = dataset.loc[:, dataset.columns != 'ttf']
    X_cols = X.columns
    X = np.array(X)
    y = dataset['ttf']

    #STAGE 1: select percentile
    indices_mask = SelectPercentile(mutual_info_regression, percentile = percentile).fit(X, y).get_support()
    indices_left = [i for i in range(len(indices_mask)) if indices_mask[i] == True]
    X_new = X[:,tuple(indices_left)]
    X_new_cols = X_cols[indices_mask]

    #STAGE 2: Recursive Feature elimination
    selector = RFE(estimator, num_of_features, step=1)
    selector = selector.fit(X_new, y)
    indices_mask =  selector.support_
    indices_left = [i for i in range(len(indices_mask)) if indices_mask[i] == True]
    X_remaining = X_new[:,tuple(indices_left)]
    X_remaining_cols = X_new_cols[indices_mask]

    purged_dataset = pd.DataFrame(X_remaining, columns=X_remaining_cols)

    return purged_dataset


if __name__ == '__main__':

    #Creating a dataset of features
    data_path = '/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/EQs/EQ_2.h5'
    df = pd.read_hdf(data_path, key='table')
    df.columns = ['s', 'ttf']
    config_path = "/home/pavel/Documents/0Research/Projects/LANL-Earthquake/src/data_processing/dp_config.json"
    import sys
    import json
    sys.path.insert(0, '/home/pavel/Documents/0Research/Projects/LANL-Earthquake/src/data_processing')
    from dp_utils import process_df

    df = pd.read_hdf(data_path, key='table')
    df.columns = ['s', 'ttf']
    with open(config_path, 'rb') as config:
        params = json.load(config)

    routines = params['routines']
    default_window_size = params['window_size']

    df = process_df(df, routines, default_window_size)


    # Working Example
    params = dict()
    params["percentile"] = 70
    params['num_of_features'] = 2
    estimator = DecisionTreeRegressor()

    new_df = eliminate_features(df, estimator, **params)

    print(new_df.columns)


