import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold

from ast import literal_eval
from folds import *

import src.validation.ValidationProc

import os
import sys
import glob

# import importlib.util
# spec = importlib.util.spec_from_file_location("utils", os.path.join(os.getcwd(), 'src/utils.py'))
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)

from src.utils import str_to_class

from tqdm import tqdm

if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    '''
    1. take the data
    2. create folds
    3. create metrics
    3. create wrappers for model or models (validation instances-vi)
    4. train vi and create summary for each vi (or each models in vi)
    '''

    # 1. path to data
    train_data = kwargs['train_data']

    # path to summary
    summary_dest = kwargs['summary_dest']

    print(' work with next Fold  objects: KFold, RepeatedKFold, LeaveOneOut, StraifiedKFold,  RepeatedStraifiedKKFold')

    # 2. parse params and create a chain of folds
    folds_list = []
    fold_features = [{}]

    for index_f, f in enumerate(kwargs['folds']):
        fold_features[index_f] = {'fold_name': f['name'], "fold_param":[]}
        class_ = str_to_class('sklearn.model_selection', f['name'])
        del f['name']
        fold_features[index_f]["fold_param"] = f
        fold_features.append({})
        fold_obj = class_(**f)
        folds_list.append(fold_obj)

    # 3. load metrics from config
    metrics = []
    for m in kwargs["metrics"]:
        metrics.append(m)

    metrics_classes = check_metrics(metrics)

    # 4. parse params and create a chain of validation instances
    validators = []
    for v in kwargs['validate']:
        class_ = str_to_class('ValidationProc', 'ValidationBase')
        validator = class_(**v)
        validators.append(validator)

    # Load data
    train_df = pd.read_hdf(train_data,key='table')
    # 4. train validators
    print('....................... Train models ..............................')

    for i_f, f in enumerate(folds_list):
        for v in tqdm(validators):
            # train models in validator and create summary for all models
            v.train_models(train_df.drop(['time_to_failure'], axis=1), train_df['time_to_failure'], f, summary_dest, metrics_classes, fold_features[i_f])

    print('.......................Processing finished.........................')




def info():
    print('Config example: validation_config.json')
    print('Example of custom model: models.py')
    print("For custom models: ")
    print("1. Function must have method predict(train_data) that return y_predict")
    print("2. Function must have method fit(train_data, train_y))")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--i', dest = "info", help = 'print info about usage of script' , action='store_true', default=False)

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str)

    args = parser.parse_args()

    if args.info:
        info()
    else:
        with open(args.config_fname) as config:
            params = json.load(config)
        main(**params)