import numpy as np
import pandas as pd
import functools

from scipy import interpolate
import inspect
import sys
import json
import os
from tqdm import tqdm
import warnings
from os import listdir
from os.path import isfile, join


class DumpDecorator:
    def __init__(self, f):
        self._func = f
        functools.update_wrapper(self, f)

    def __call__(self, target, *args, **kwargs):
        df = self._func(target, *args, **kwargs)

        if "save_path" in kwargs:
            tqdm.write("\t dump decorator: ")
            tqdm.write("\t - save_path: {}".format(kwargs["save_path"]))
            df.to_hdf(kwargs["save_path"], key='table')
        return df


class DataFrameDecorator:
    def __init__(self, f):
        self.unwrapped = f
        functools.update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        df = args[0]
        if "desc_line" in kwargs:
            desc_line = kwargs['desc_line']
        else:
            desc_line = get_function_descriptor(self.unwrapped, kwargs)

        tqdm.write(f"{desc_line}:")
        tqdm.write("\t DataFrame decorator: wraps the output in pandas DataFrame")
        temp = self.unwrapped(df.values, *args, **kwargs)
        return pd.DataFrame(temp, columns={desc_line})


class WindowDecorator:
    def __init__(self, f):
        self.unwrapped = f
        functools.update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        if self.unwrapped is None:
            self.unwrapped = args[0]

        df = args[0]
        window_size = kwargs['window_size']
        if "desc_line" in kwargs:
            desc_line = kwargs['desc_line']
        else:
            desc_line = get_function_descriptor(self.unwrapped, kwargs)

        # TODO: fix this hack
        if window_size >= df.shape[0]:
            tqdm.write(f"{desc_line}:")
            tqdm.write("\t window decorator: ")
            tqdm.write("\t - window size: {}".format(window_size))
            temp = self.unwrapped(df.values, *args, **kwargs)
            return pd.DataFrame(temp, columns={desc_line})
        else:
            temp = []
            for i in tqdm(range(0, df.shape[0], window_size),
                          desc=desc_line, file=sys.stdout):
                batch = df.iloc[i: i + window_size].values
                temp.append(self.unwrapped(batch, *args, **kwargs))
            tqdm.write("\t window decorator: ")
            tqdm.write("\t - window size: {}".format(window_size))
            if hasattr(temp[0], 'shape') and temp[0].shape is not ():
                out_features = temp[0].shape[0]
                column_names = [desc_line + '_' + str(i) for i in range(out_features)]
            else:
                column_names = [desc_line]

            return pd.DataFrame(temp, columns=column_names)


def get_function_descriptor(func, extra_params):

    """
    Parse all function args and get a complete description

    Parameters
    ----------
    func : function reference object
    extra_params : external params passed to decorated function (i.e. window size, desc_line, etc...)

    Returns
    -------
    String with function name and its parameters
    """

    inspect_params = inspect.getfullargspec(func)
    funcname_line = func.__name__
    args_line = "{}, ".format(*inspect_params.args)
    kwargs_line = ', '.join("{}={}".format(k, v) for k, v in extra_params.items())
    if inspect_params.kwonlydefaults is not None:
        kwonlydefaults_line = ', (' + ', '.join("{}={}".format(k, v) for k, v in inspect_params.kwonlydefaults.items()) \
                              + ')'
    else:
        kwonlydefaults_line = ''
    
    desc_line = funcname_line + '(' + args_line + kwargs_line + kwonlydefaults_line + ')'

    return desc_line


def process_df(df, routines, default_window_size, df_path=None):
    """
    Data processing is done in three main steps:
    1) Calculate all features listed in configuration file (dp_config.json)
    2) Append labels
    3) Perform data resampling in case different window sizes are used

    :param df: raw data
    :type df: pandas DataFrame
    :param routines: list of routines specified in dp_config.json file
    :type routines: list
    :param default_window_size:
    :type default_window_size:
    :return: dataframe with features calculated based on the list of routines
    :rtype: pandas DataFrame
    """
    import dp_features
    temp_data = {}
    processed_features = []

    # Create dir with df name if it doesn't exist
    if df_path is not None:
        if os.path.exists(df_path):
            processed_features = [os.path.splitext(f)[0] for f in listdir(df_path) if isfile(join(df_path, f))]
            warnings.warn(f"Directory {df_path} already exists. "
                          f"Running data processing in this directory again might lead to data loss.")
        else:
            os.makedirs(df_path)

    # calc all features
    print(f"Calculation of {len(routines)} features...")
    for routine in routines:
        if not routine['on']:
            print(f"Feature {routine['name']} calcualtion is disabled")
            continue

        func = getattr(dp_features, routine["name"])
        func_params = routine['params']
        window_size = default_window_size if 'window_size' not in routine else routine['window_size']
        if func_params:
            desc_line = f"{func.__name__}({routine['column_name']}, window_size={window_size}, " + \
                        ', '.join("{!s}={!r}".format(key, val) for (key, val) in func_params.items()) + ')'
        else:
            desc_line = f"{func.__name__}({routine['column_name']}, window_size={window_size})"

        if desc_line in processed_features:
            print(f"File {desc_line}.h5 already exists. Skip calculations and append it to the dataframe")
            data_processed = pd.read_hdf(os.path.join(df_path, desc_line + ".h5"), key='table')
            new_col_name = list(data_processed)[0]
            temp_data[new_col_name] = data_processed
            continue

        try:
            data = df[routine['column_name']] if routine['column_name'] in df.columns \
                else temp_data[routine['column_name']].squeeze()
        except KeyError as e:
            raise KeyError(f"Check your feature calculation order, key: {e} is missing")

        save_path = os.path.join(df_path, desc_line + ".h5")
        data_processed = func(data,
                              window_size=window_size,
                              desc_line=desc_line,
                              save_path=save_path,
                              **func_params)
        new_col_name = data_processed.columns.values.tolist()[0]
        temp_data[new_col_name] = data_processed

    # append column with labels
    try:
        temp_data['ttf'] = dp_features.w_last_elem(df['ttf'], window_size=default_window_size, desc_line="ttf")
    except KeyError as e:
        raise KeyError(f"Labels can't be calculated, key: {e} is missing")

    # perform resampling if needed
    resulted_size = temp_data['ttf'].shape[0]
    for k, v in temp_data.items():
        temp_data[k] = resample_column(v, resulted_size)

    res = pd.concat(temp_data.values(), axis=1)
    return res


def resample_column(df, resulted_size):
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
        col_name = list(df)[0]

        if old_size < resulted_size:
            print(f'upsample {col_name} from {old_size} to size {resulted_size}')
        else:
            print(f'downsample {col_name} from {old_size} to size {resulted_size}')

        x_old = np.arange(0, old_size)
        f = interpolate.interp1d(x_old, df[col_name])
        x_new = np.linspace(0, old_size - 1, resulted_size)
        y_new = f(x_new)
        return pd.DataFrame(y_new, columns={col_name})
