import json
import os
import argparse
import pprint
import platform
import pandas as pd
import numpy as np
import matplotlib as mpl
import pickle
import copy
from collections import defaultdict

from src.utils import str_to_class
from src.validation.summary_utils import summarize
from src.validation.validation_utils import read_write_summary

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

    verbose = kwargs['verbose']
    # 0. Create folders for train/validation images
    img_folder_path = kwargs["img_folder_path"]
    folder_nums = np.arange(0, 5, 1)
    for folder_num in folder_nums:
        os.makedirs(os.path.join(img_folder_path, str(folder_num)), exist_ok=True)

    # 1. load data
    train_data_fname = kwargs['train_data_fname']
    train_df = pd.read_hdf(train_data_fname, key='table')

    # train_df
    train_data = train_df.drop(['ttf'], axis=1)
    y_train_data = train_df['ttf']

    # 2. create preprocessor
    preproc_cls = None
    if 'preproc' in kwargs:
        preproc_kwargs = copy.deepcopy(kwargs['preproc'])
        del preproc_kwargs['name']
        preproc_cls = str_to_class("src.preprocessing.preproc", kwargs['preproc']['name'])

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
    if verbose:
        print("Model params:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(model_params)
    if 'name' in model_params:
        model_params.pop("name")

    # 6. train model
    if verbose:
        print('....................... Training model ..............................')
    scores = defaultdict(list)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
        print(f" --------------- fold #{fold_n} -------------------")
        # split data
        X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
        y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]

        if preproc_cls is not None:
            preprocessor = preproc_cls(**preproc_kwargs)
            preprocessor.fit(X_train)
            X_train = pd.DataFrame(preprocessor.transform(X_train))
            X_valid = pd.DataFrame(preprocessor.transform(X_valid))

        model = model_cls(**model_params)

        model.fit(X_train, y_train)

        # validate
        predict = model.predict(X_train)
        for metric_name, metric in metrics.items():
            score = metric(predict, y_train)
            scores[metric_name+'_train'].append(score)
            if verbose:
                print(f"train score ({metric_name}): {score.mean():.4f}")

        if X_valid.shape[0] != 0:
            predict = model.predict(X_valid)
            for metric_name, metric in metrics.items():
                score = metric(predict, y_valid)
                scores[metric_name].append(score)
                if verbose:
                    print(f"validation score ({metric_name}): {score.mean():.4f}")

        train_data_scaled = pd.DataFrame(preprocessor.transform(train_data))
        predict = model.predict(train_data_scaled)
        plt.figure(figsize=(100, 5), dpi=300)
        plt.plot(predict, 'k')
        plt.plot(y_train_data, 'r')
        plt.plot(valid_index, model.predict(X_valid), 'b')
        plt.title(f"V: {scores['mean_absolute_error'][-1]} | T:{scores['mean_absolute_error_train'][-1]}")
        plt.grid(True)
        plot_path = os.path.join(img_folder_path,
                                 f"{np.digitize(scores['mean_absolute_error'][-1], folder_nums) - 1}",
                                 f"v-{scores['mean_absolute_error'][-1]:.4f}__t-{scores['mean_absolute_error_train'][-1]:.4f}")
        plt.savefig(f"{plot_path}.png")

    # save last model
    # TODO: consider saving the best performing model instead of the last one
    with open(f"Model {kwargs['model']['name']}", 'wb') as file:
        pickle.dump(model, file)

    # 7. create summary
    summary_dest_fname = kwargs['summary_dest_fname']
    if verbose:
        print(f'Add summary to "{summary_dest_fname}"')
    summary_row = summarize(scores=scores, **kwargs)

    _, summary_ext = os.path.splitext(summary_dest_fname)
    if not os.path.exists(summary_dest_fname):
        read_write_summary(summary_dest_fname, summary_ext, 'wb', summary_row)
    else:
        # load summary
        summary = read_write_summary(summary_dest_fname, summary_ext, 'rb')
        # concatenate load summary and new model's summary
        summary = pd.concat((summary, summary_row), sort=False).fillna(np.nan) if summary is not None else summary_row
        read_write_summary(summary_dest_fname, summary_ext, '+wb', summary)

    if verbose:
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
