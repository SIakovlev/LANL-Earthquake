import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold


from folds import *


# from ..src.utils import str_to_class


if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    train_data = kwargs['train_data']
    test_data = kwargs['test_data']
    summary_dest = kwargs['summary_dest']
    print(' work with next Fold  objects: KFold, RepeatedKFold, LeaveOneOut, StraifiedKFold,  RepeatedStraifiedKKFold')

    # 1. Parse params and create a chain of folds
    folds_list = []
    for f in kwargs['folds']:
        class_ = str_to_class('sklearn.model_selection',f['name'])
        del f['name']
        fold_obj = class_(**f)
        folds_list.append(fold_obj)

    # 2. Parse params and create a chain of validation instances
    validators = []

    for v in kwargs['validate']:
        class_ = str_to_class('ValidationProc', v['name'])
        validator = class_(**v)
        validators.append(validator)

    # 2. Load data
    #TODO: implement "smart" data loading to handle data too big to fit in the memory
    train_df = pd.read_hdf(train_data,key='table')
    test_df = pd.read_hdf(test_data, key='table')
    #
    print(train_df.columns)
    # 3. Run train
    print('.......................Processing started.........................')

    for f in folds_list:
        for i, v in enumerate(validators):
            #print(f'{i}: fold_type = {f.__class__.__name__} | name={v.__class__.__name__} | models={v.models_list }')
            v.train_model(train_df.drop(['time_to_failure'], axis=1), train_df['time_to_failure'], f)
            # v.summary(summary_dest)
    # # 4. Save modified dataframe
    # processors[0].save(df, data_fname_dest)
    # print(f'dataframe saved as{data_fname_dest}')
    #
    # pd.set_option('display.max_columns', 500)
    # print('.......................Processing finished.........................')
    # print(df.tail(10))

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