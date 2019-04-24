import json
import argparse
import platform
import pandas as pd
import numpy as np
import matplotlib as mpl
import pickle
import copy
from collections import defaultdict

from src.utils import str_to_class
from src.validation.summary_utils import summarize

import xgboost as xgb

from src.models.mlp_net import MLP
from src.models.mlp_classifier_net import MLP_classifier
from src.models.lstm_net import LstmNet
from src.folds.folds import CustomFold
from src.models.nn_test import CustomNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific

import matplotlib.pyplot as plt

from src.models.mlp_net import MLP


def main(**kwargs):
    '''
    1. take the data
    2. create preprocessor
    3. create folds
    4. create metrics
    5. create model
    6. train model
    7. save summary
    '''

    # 1. load data
    train_data_fname = kwargs['train_data_fname']
    train_df = pd.read_hdf(train_data_fname, key='table')

    # train_df
    train_data = train_df.drop(['ttf'], axis=1)
    y_train_data = train_df['ttf']

    # path to summary
    summary_dest_fname = kwargs['summary_dest_fname']
    try:
        with open(summary_dest_fname, 'rb') as f:
            summary = pickle.load(f)
    except:
        print(f"Couldn't open summary file: {summary_dest_fname}")
        summary = None

    # 2. create preprocessor
    # TODO: add Standard Scaler
    preprocessor = None
    if 'preproc' in kwargs:
        preprocessor = str_to_class("src.preprocessing.preproc", kwargs['preproc']['name'])()

    # 3. create folds
    folds_kwargs = copy.deepcopy(kwargs['folds'])
    del folds_kwargs['name']
    try:
        # first look for folds in sklearn
        folds = str_to_class('sklearn.model_selection', kwargs['folds']['name'])(**folds_kwargs)
    except (AttributeError, ImportError) as e:
        try:
            # then try to look among imported folds
            folds = str_to_class(__name__, kwargs['folds']['name'])(**folds_kwargs)
        except (AttributeError, ImportError) as e1:
            raise e1

    # 4. create metrics
    metrics_classes = [str_to_class('sklearn.metrics', m) for m in kwargs['metrics']]
    metrics = dict(zip(kwargs['metrics'], metrics_classes))

    # 5. create model
    model_cls = str_to_class(__name__, kwargs['model']['name'])

    # instantiate and train model
    model_params = copy.deepcopy(kwargs['model'])
    if 'name' in model_params:
        model_params.pop("name")

    # 6. train model
    print('....................... Training model ..............................')
    scores = defaultdict(list)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
        print(f" --------------- fold #{fold_n} out of -------------------")
        # split data
        X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
        y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]

        preprocessor.fit(X_train)
        X_train = pd.DataFrame(preprocessor.transform(X_train))
        X_valid = pd.DataFrame(preprocessor.transform(X_valid))

        model = model_cls(**model_params)

        model.fit(X_train, y_train)

        # validate

        rand_train_idx = np.random.randint(0, X_train.shape[0], X_valid.shape[0])
        predict = model.predict(X_train.iloc[rand_train_idx])
        for metric_name, metric in metrics.items():
            score = metric(predict, y_train.iloc[rand_train_idx])
            scores[metric_name].append(score)
            print(f"train score ({metric_name}): {score.mean():.4f}")

        predict = model.predict(X_valid)
        for metric_name, metric in metrics.items():
            score = metric(predict, y_valid)
            scores[metric_name].append(score)
            print(f"validation score ({metric_name}): {score.mean():.4f}")

        # predict = model.predict(X_valid)
        # plt.figure(figsize=(10, 5))
        # plt.plot(predict, 'k')
        # plt.plot(y_valid.values, 'r')
        # plt.show()


    # save last model
    # TODO: consider saving the best performing model instead of the last one
    with open(f"Model {kwargs['model']['name']}", 'wb') as file:
        pickle.dump(model, file)

    # 7. create summary
    summary_row = summarize(scores=scores, **kwargs)
    summary = pd.concat((summary, summary_row)).fillna(np.nan) if summary is not None else summary_row
    with open(summary_dest_fname, 'wb') as f:
        pickle.dump(summary, f)
    print(summary_row.columns.values)
    print(summary_row.head(1))

    print('.......................Training finished.........................')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--i', dest="info", help='print info about usage of script', action='store_true', default=False)

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str,
                        default="../configs/train_config.json")

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)
