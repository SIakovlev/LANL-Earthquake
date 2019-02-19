import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold


from folds import *




if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    train_data = kwargs['train_data']
    summary_dest = kwargs['summary_dest']
    print(' work with next Fold  objects: KFold, RepeatedKFold, LeaveOneOut, StraifiedKFold,  RepeatedStraifiedKKFold')

    # 1. Parse params and create a chain of folds
    folds_list = []
    for f in kwargs['folds']:
        class_ = str_to_fold_object(f['name'])
        del f['name']
        fold_data = class_(**f)
        folds_list.append(fold_data)

    # 2. Parse params and create a chain of validation instances
    validators = []

    for v in kwargs['validate']:
        class_ = str_to_class('ValidationProc', v['name'])
        validator = class_(**v)
        validators.append(validator)

    # 2. Load data
    #TODO: implement "smart" data loading to handle data too big to fit in the memory
    df = pd.read_hdf(train_data,key='table')
    #

    # 3. Run train
    print('.......................Processing started.........................')

    for f in folds_list:
        for i, v in enumerate(validators):
            print(f'{i}: fold_type = {f.__class__.__name__} | name={v.__class__.__name__} | model={v.model}')
            #v.train_model()
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