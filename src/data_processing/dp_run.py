import os
import json
import argparse
import inspect
import dp_utils as dp
import pandas as pd
import functools
from tqdm import tqdm


def main(**kwargs):
    data_fname = kwargs['data_path']
    data_fname_dest = kwargs['data_processed_path']

    # # 1. Parse params and create a chain of processing instances
    func_list = []
    for name in list(kwargs['routines'].keys()):
        func_list.append(getattr(dp, name))

    # 2. Load data
    df = pd.read_hdf(data_fname, key='table')

    # 3. Run processing
    routine_settings = list(kwargs['routines'].values())

    dfp = pd.concat(
        [func(df['s'], **setting['params']) for func, setting in zip(func_list, routine_settings) if setting['on']],
        axis=1)

    dfp = dfp.join(dp.w_labels(df['ttf']))

    # 4. Save modified dataframe
    dfp.to_hdf(data_fname_dest, key='table')
    # processors[0].save(df, data_fname_dest)
    print(f'Processed dataframe saved at {data_fname_dest}')

    pd.set_option('display.max_columns', 500)
    print('.......................Processing finished.........................')
    print(dfp.head(10))


if __name__ == '__main__':

    config_fname = "dp_config.json"
    # build config if there is no .json file
    if not os.path.isfile(config_fname):
        func_ref_list = [obj[1] for obj in inspect.getmembers(dp) if inspect.isfunction(obj[1])]

        # Sergey Ubuntu PC paths
        dp_config = {"data_path": "/home/sergey/Projects/Kaggle/LANL-Earthquake-Prediction/train/train.h5",
                     "data_processed_path": "/home/sergey/Projects/Kaggle/LANL-Earthquake-Prediction/train/train_processed.h5",
                     "window_size": 10000,
                     "routines": {}}

        # Sergey Mac OS paths
        # dp_config = {"data_path": "/Users/sergey/Dev/Kaggle/LANL-Earthquake-Prediction/train/train.h5",
        #              "data_processed_path": "/Users/sergey/Dev/Kaggle/LANL-Earthquake-Prediction/train/train_processed.h5",
        #              "window_size": 10000,
        #              "routines": {}}

        for obj in func_ref_list[:-1]:
            inspect_obj = inspect.signature(obj)
            params_dict = dict(inspect_obj.parameters)
            params = {}
            for k, v in params_dict.items():
                if v.default != inspect._empty:
                    params[k] = v.default
            dp_config["routines"][obj.__name__] = {"on": True, "column_name": "s", "params": params}

        with open(config_fname, 'w') as outfile:
            json.dump(dp_config, outfile, indent=2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str,
                        default=config_fname)

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)


def update_config(name):
    pass
    # func_ref_list = [obj[1] for obj in inspect.getmembers(dp) if inspect.isfunction(obj[1])]
    #
    # # Sergey Ubuntu PC paths
    # dp_config = {"data_path": "/home/sergey/Projects/Kaggle/LANL-Earthquake-Prediction/train/train.h5",
    #              "data_processed_path": "/home/sergey/Projects/Kaggle/LANL-Earthquake-Prediction/train/train_processed.h5",
    #              "window_size": 10000,
    #              "routines": {}}

    # Sergey Mac OS paths
    # dp_config = {"data_path": "/Users/sergey/Dev/Kaggle/LANL-Earthquake-Prediction/train/train.h5",
    #              "data_processed_path": "/Users/sergey/Dev/Kaggle/LANL-Earthquake-Prediction/train/train_processed.h5",
    #              "window_size": 10000,
    #              "routines": {}}

    # for obj in func_ref_list[:-1]:
    #     inspect_obj = inspect.signature(obj)
    #     params_dict = dict(inspect_obj.parameters)
    #     params = {}
    #     for k, v in params_dict.items():
    #         if v.default != inspect._empty:
    #             params[k] = v.default
    #     dp_config["routines"][obj.__name__] = {"on": True, "column_name": "s", "params": params}
    #
    # with open(name, 'w') as outfile:
    #     json.dump(dp_config, outfile, indent=2)

    # TODO: change def valus of functions if needed based on config file
    # functools.partial(obj.__name__, **params)