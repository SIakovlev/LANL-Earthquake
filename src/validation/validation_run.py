import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
import copy

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

from ast import literal_eval
import os
import sys

sys.path.append(os.getcwd())


import src.validation.validation_utils
from src.folds.folds import CustomFold
import warnings
warnings.filterwarnings("ignore")


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


    optional_save = {'models_directory':None, 'predict_directory':None}

    # path to save models and output_predict(optional)
    for opt_elem_key, _ in optional_save.items():
        if opt_elem_key in kwargs:
            optional_save[opt_elem_key] = kwargs[opt_elem_key]
            #check if the directory exist
            if not os.path.exists(optional_save[opt_elem_key]):
                os.makedirs(optional_save[opt_elem_key])

    # Load data
    print(f' - Attempt to load data from {train_data}')
    train_df = pd.read_hdf(train_data, key='table')
    train_columns= copy.copy(train_df.columns)
    y_data = train_df[train_columns[-1]]
    train_df = train_df.drop([train_columns[-1]], axis=1)

    # 2. parse params and create a chain of preprocessors
    preprocessor = None
    if 'preproc' in kwargs:
        print(kwargs['preproc']['name'])
        name_class = kwargs['preproc']['name']
        del kwargs['preproc']['name']
        preprocessor  = {"preproc_name":name_class, "preproc_params":kwargs['preproc']}

    # 3. parse params and create a chain of folds
    folds_list = []
    fold_features = [{}]

    for index_f, f in enumerate(kwargs['folds']):
        fold_features[index_f] = {'folds_name': f['name'], "folds_params":[]}
        class_ = str_to_class(__name__, f['name'])
        del f['name']
        fold_features[index_f]["folds_params"] = f
        fold_features.append({})
        fold_obj = class_(**f)
        folds_list.append(fold_obj)

    # 4. load metrics from config
    metrics_classes = {k: str_to_class('sklearn.metrics', k) for k in kwargs["metrics"]}

    # 5. parse params and create a chain of validation instances
    validators = []
    for v in kwargs['validate']:
        class_ = str_to_class('validation_utils', 'ValidationBase')
        validator = class_(**v)
        validators.append(validator)


    # 4. train validators
    print('....................... Train models ..............................')
    print(train_df)
    for i_f, f in enumerate(folds_list):
        for v in validators:
            # train models in validator and create summary for all models
            v.train_models(train_df,
                           y_data,
                           f,
                           summary_dest,
                           metrics_classes,
                           fold_features[i_f],
                           preprocessor,{'data_fname':train_data}, optional_save)

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
                        type=str,
                        default='../configs/validation_config.json')

    args = parser.parse_args()

    if args.info:
        info()
    else:
        with open(args.config_fname) as config:
            params = json.load(config)
        main(**params)
