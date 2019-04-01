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

from src.models.nn_test import CustomNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

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
    preprocessor = None
    if 'preproc' in kwargs:
        preprocessor = str_to_class("src.validation.preproc", kwargs['preproc']['name'])(**kwargs['preproc'])

    # 3. create folds
    folds_kwargs = copy.deepcopy(kwargs['folds'])
    del folds_kwargs['name']
    folds = str_to_class('sklearn.model_selection', kwargs['folds']['name'])(**folds_kwargs)

    # 4. create metrics
    metrics_classes = [str_to_class('sklearn.metrics', m) for m in kwargs['metrics']]
    metrics = dict(zip(kwargs['metrics'], metrics_classes))

    # 5. create model
    model_cls = str_to_class(__name__, kwargs['model']['name'])

    # instantiate and train model
    model_params = copy.deepcopy(kwargs['model'])
    # TODO: fix this
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

        plt.figure(figsize=(30,15))
        plt.imshow(np.log(X_valid+1.0).T, vmax=0.001)
        plt.savefig(f'valid_data_log_{fold_n}.png')
        plt.figure(figsize=(30,15))
        plt.imshow(X_valid.T, vmax=0.001)
        plt.savefig(f'valid_data_{fold_n}.png')


        model = model_cls(**model_params)
        model.fit(X_train, y_train)

        # validate

        # temp custom reshape data for
        num_samples_to_show = 20000

        X_train_reshaped = pd.DataFrame(np.stack([X_train[i-15:i] for i in range(15, X_train.shape[0])]).reshape(-1, 30000)[:num_samples_to_show])


        predict = model.predict(X_train_reshaped)

        plt.figure()
        plt.plot(y_train.values)
        plt.plot(predict)
        plt.ylim([0., 20.])
        plt.title('train')
        # plt.show(block=False)
        plt.savefig(f'train_{fold_n}.png')

        for metric_name, metric in metrics.items():
            score = metric(predict, y_train[15: num_samples_to_show + 15])
            scores[metric_name].append(score)
            print(f"train score: {score.mean():.4f}")

        # temp custom reshape data for
        X_valid_reshaped = pd.DataFrame(np.stack([X_valid[i - 15:i] for i in range(15, X_valid.shape[0])]).reshape(-1, 30000))

        predict = model.predict(X_valid_reshaped)

        plt.figure()
        plt.plot(y_valid.values)
        plt.plot(predict)
        plt.ylim([0., 20.])
        plt.title('valid')
        # plt.show(block=False)
        plt.savefig(f'valid_{fold_n}.png')

        for metric_name, metric in metrics.items():
            score = metric(predict, y_valid[15:])
            scores[metric_name].append(score)
            print(f"validation score: {score.mean():.4f}")

    # with open(f"Model {kwargs['model']['name']}", 'wb') as file:
    #     pickle.dump(model, file)

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

    parser.add_argument('--i', dest = "info", help = 'print info about usage of script' , action='store_true', default=False)

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str,
                        default="../configs/mlp_train_config.json")

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)
