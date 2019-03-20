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

    default_window_size = kwargs['window_size']

    # 1. Load data
    df = pd.read_hdf(data_fname, key='table')

    # 2. Run processing
    dfp = dp.process_df(df, kwargs['routines'], default_window_size)

    # 3. Save modified dataframe
    dfp.to_hdf(data_fname_dest, key='table')

    print(f'Processed dataframe saved at {data_fname_dest}')
    pd.set_option('display.max_columns', 500)
    print('.......................Processing finished.........................')
    print(dfp.head(10))


if __name__ == '__main__':

    config_fname = "dp_config.json"
    # build config if there is no .json file
    if not os.path.isfile(config_fname):
        func_ref_list = [obj[1] for obj in inspect.getmembers(dp) if obj[0].startswith("w_")]

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

        routines = []
        for obj in func_ref_list[:-1]:
            inspect_obj = inspect.signature(obj)
            params_dict = dict(inspect_obj.parameters)
            params = {}
            for k, v in params_dict.items():
                if v.default != inspect._empty:
                    params[k] = v.default
            routines.append({"name": obj.__name__, "on": True, "column_name": "s", "params": params})
        dp_config["routines"] = routines
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


# TODO: add in future if needed
def update_config(name):
    pass
