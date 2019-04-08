import os
import json
import argparse
import inspect
import dp_utils as dp
# import dp_features
import pandas as pd
import platform


def main(**kwargs):
    data_path = kwargs['data_dir'] + kwargs["data_fname"]
    data_processed_path = kwargs['data_processed_dir'] + kwargs["data_processed_fname"]

    default_window_size = kwargs['window_size']
    default_window_stride = kwargs['window_stride']

    # 1. Load data
    print('.......................Processing started.........................')
    print(f' - Attempt to load data from {data_path}')
    df = pd.read_hdf(data_path, key='table')
    print(' - Data was successfully loaded into memory')

    # 2. Run processing
    print(' - Run dataframe processing')
    dfp = dp.process_df(df,
                        kwargs['features'],
                        default_window_size,
                        default_window_stride,
                        kwargs['data_processed_dir'] + os.path.splitext(kwargs["data_processed_fname"])[0] + '/')
    print(' - Dataframe was successfully processed')

    # 3. Save modified dataframe

    print(f' - Attempt to save modified data to {data_processed_path}')
    dfp.to_hdf(data_processed_path, key='table')
    print(f' - Processed dataframe saved at {data_processed_path}')

    pd.set_option('display.max_columns', 500)
    print('.......................Processing finished.........................')
    print(dfp.head(10))


if __name__ == '__main__':

    config_fname = "../configs/dp_config.json"
    # build config if there is no .json file
    if not os.path.isfile(config_fname):
        # MacOS specific
        if platform.system() == 'Darwin':
            dp_config = {"data_dir": "../../data.nosync/",
                         "data_processed_dir": "../../data.nosync/"}
        else:
            dp_config = {"data_dir": "../../data/",
                         "data_processed_dir": "../../data/"}
        dp_config.update({"data_fname": "train.h5",
                          "data_processed_fname": "train_processed.h5",
                          "window_size": 150000,
                          "window_stride": 1000,
                          "features": {}})

        # Create features dict with a feature example
        features = [{"name": "example",
                     "on": True,
                     "functions": {
                         "r_std": {"window_size": 100, "window_stride": None},
                         "w_quantile": {"q": 0.05}
                     }}]

        dp_config["features"] = features
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
