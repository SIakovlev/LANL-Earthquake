import os
import warnings
import json
import math
import numpy as np
import pandas as pd
from feature import Feature

from scipy import interpolate


def process_df(df, features, default_window_size, default_window_stride, df_path=None):

    """
    Data processing is done in three main steps:
    1) Calculate all features listed in configuration file (dp_config.json)
    2) Append labels
    3) Perform data re-sampling in case different window sizes are used
    :param df: raw data pandas DataFrame
    :param features: list of features specified in .json file
    :param default_window_size:
    :param default_window_stride:
    :param df_path: path to a folder where the processed pandas DataFrame will be stored
    :param is_test: test/non-test dataset flag (True in case if train dataset is used)
    :return: processed pandas DataFrame
    """

    temp_data = {}

    # Create dir with df name if it doesn't exist
    if df_path is not None:
        if os.path.exists(df_path):
            warnings.warn(f"Directory {df_path} already exists. "
                          f"Running data processing in this directory again might lead to data loss.")
        else:
            os.makedirs(df_path)

    # calc all features
    print(f"Calculation of {len(features)} features...")
    general_params = (default_window_size, default_window_stride, df_path)
    for feature in features:
        if not feature['on']:
            print(f"Feature {feature['name']} calcualtion is disabled")
            continue
        if 'ttf' in df.columns:
            data_processed, feature_name = calculate_feature(df.drop(['ttf'], axis=1), feature, *general_params)
        else:
            data_processed, feature_name = calculate_feature(df, feature, *general_params)
        temp_data[feature_name] = data_processed

    if 'ttf' in df.columns:
        # calculate column with labels for the processed dataframe
        try:
            temp_data['ttf'] = Feature(df=df['ttf'], save_dir=df_path).w_last_elem(df['ttf'],
                                                                                   window_size=default_window_size,
                                                                                   window_stride=default_window_stride,
                                                                                   desc_line="ttf").data
        except KeyError as e:
            raise KeyError(f"Labels can't be calculated, key: {e} is missing")

    # perform resampling if needed
    resulted_size = math.ceil((df.shape[0] - default_window_size + default_window_stride) / default_window_stride)
    for k, v in temp_data.items():
        temp_data[k] = resample_columns(v, resulted_size)

    return pd.concat(temp_data.values(), axis=1)


def calculate_feature(df, feature, default_window_size, default_window_stride, save_dir):

    feature_obj = Feature(df=df, save_dir=save_dir)

    for func_name, func_params in feature['functions'].items():
        window_size = default_window_size if 'window_size' not in func_params \
            else func_params['window_size']
        window_stride = default_window_stride if 'window_stride' not in func_params \
            else func_params['window_stride']
        func_params["window_size"] = window_size
        func_params["window_stride"] = window_stride

        desc_line = f"{func_name}(self, " + \
                    ', '.join("{!s}={!r}".format(key, val) for (key, val) in func_params.items()) + ')'
        func_params["desc_line"] = desc_line
        feature_obj = getattr(feature_obj, func_name)(**func_params)
        feature_obj = feature_obj.dump()

    return feature_obj.data, feature_obj.get_name()


def calculate_feature_by_name(df, feature_name, save_dir, config_name='../configs/dp_config.json'):

    with open(config_name) as config:
        params = json.load(config)

    default_window_size = params['window_size']
    default_window_stride = params['window_stride']
    features = params["features"]
    print(features)
    for obj in features:
        print(obj['name'])
        if obj['name'] == feature_name:
            feature = obj
            break

    feature_obj = Feature(df=df, save_dir=save_dir)

    for func_name, func_params in feature['functions'].items():
        window_size = default_window_size if 'window_size' not in func_params \
            else func_params['window_size']
        if window_size is None:
            window_stride = None
        else:
            window_stride = default_window_stride if 'window_stride' not in func_params \
                else func_params['window_stride']
        func_params["window_size"] = window_size
        func_params["window_stride"] = window_stride
        desc_line = f"{func_name}(self, " + \
                    ', '.join("{!s}={!r}".format(key, val) for (key, val) in func_params.items()) + ')'
        func_params["desc_line"] = desc_line
        feature_obj = getattr(feature_obj, func_name)(**func_params)
        feature_obj = feature_obj.dump()

    return feature_obj.data, feature_obj.get_name()


def resample_columns(df, resulted_size):
    """
    Perform resampling of dataframe df to the resulted size
    Parameters
    ----------
    df : pandas DataFrame
    resulted_size (int) : size of the resampled dataframe
    Returns
    -------
    """
    old_size = df.shape[0]

    if old_size == resulted_size:
        return df
    else:
        temp = []
        col_names = list(df)

        if old_size < resulted_size:
            print(f'upsample {col_names[0]} from {old_size} to size {resulted_size}')
        else:
            print(f'downsample {col_names[0]} from {old_size} to size {resulted_size}')

        for column in df:
            x_old = np.arange(0, old_size)
            f = interpolate.interp1d(x_old, df[column])
            x_new = np.linspace(0, old_size - 1, resulted_size)
            y_new = f(x_new)
            temp.append(y_new)

        return pd.DataFrame(dict(zip(col_names, temp)), columns=col_names)
