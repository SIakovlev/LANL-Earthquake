import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold


from folds import *

import os
import sys

sys.path.append(os.path.dirname(os.path.expanduser('../src/utils.py')))

from utils import str_to_class
from tqdm import tqdm

import nn_test

if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    '''
    1. take the data
    2. create folds
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
        class_ = str_to_class('sklearn.model_selection',f['name'])
        del f['name']
        fold_features[index_f]["fold_param"] = f
        fold_features.append({})
        fold_obj = class_(**f)
        folds_list.append(fold_obj)
    # 3. load metrics from config
    metrics = []
    for m in kwargs["metrics"]:
        metrics.append(m)


    # 4. parse params and create a chain of validation instances
    validators = []
    for v in kwargs['validate']:
        class_ = str_to_class('ValidationProc', v['name'])
        validator = class_(**v)
        # check if metrics in class
        validator.check_metrics(metrics)
        validators.append(validator)


    # Load data
    train_df = pd.read_hdf(train_data,key='table')
    # 4. train models
    print('....................... Train models ..............................')

    for i_f, f in enumerate(folds_list):
        for i, v in enumerate(tqdm(validators)):
            # create the summary in summary_dest
            for i_m, m in enumerate(metrics):
                v.train_model(train_df.drop(['time_to_failure'], axis=1), train_df['time_to_failure'], f, fold_features[i_f] , m, summary_dest)

    print('.......................Processing finished.........................')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fname',
                        help='name of the config file',
                        type=str,
                        default='validation_config.json')

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)