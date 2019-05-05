import os
import json
import argparse
import dp_utils as dp
import pandas as pd
import platform
from src.validation.validation_utils import read_write_summary

def main(**kwargs):
    data_path = kwargs['data_dir'] + kwargs["data_fname"]
    data_processed_path = kwargs['data_processed_dir'] + kwargs["data_processed_fname"]

    default_window_size = kwargs['window_size']
    default_window_stride = kwargs['window_stride']

    # 1. Load data
    print('.......................Processing started.........................')
    print(f' - Attempt to load data from {data_path}')
    _, file_extension = os.path.splitext(data_path)
    df = read_write_summary(data_path,file_extension, 'rb')
    print(' - Data was successfully loaded into memory')

    # 2. Run processing
    print(' - Run dataframe processing')
    dfp = dp.process_df(df,
                        kwargs['features'],
                        default_window_size,
                        default_window_stride,
                        kwargs['data_processed_dir'] + os.path.splitext(kwargs["data_processed_fname"])[0] + '/',
                        )
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

        # Create features dict with a feature set
        win_list = [11, 101, 1001, 10001]

        features_1st = ['r_std', 'r_mean', 'r_sta_lta']

        features_2nd = [
            'w_median',
            'w_mean',
            'w_min',
            'w_max',
            'w_binned_entropy',
            'w_quantile',
            'w_kurtosis',
            'w_skewness',
            'w_q31',
            'w_ratio_beyond_r_sigma',
            'w_median_BRP',
            'w_autocorrelation',
            'w_count_above_mean',
            'w_mean_abs_change',
            'w_mean_change'
        ]

        features = []
        for win in win_list:
            for feature1 in features_1st:
                for feature2 in features_2nd:
                    feature = {"name": "", "on": True}
                    functions = {}
                    if feature1 == 'r_sta_lta':
                        functions[feature1] = {"sta_window": win, "lta_window": win * 10}
                    else:
                        functions[feature1] = {"window_size": win, "window_stride": None}
                    functions[feature2] = {}
                    feature["functions"] = functions
                    features.append(feature)

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
