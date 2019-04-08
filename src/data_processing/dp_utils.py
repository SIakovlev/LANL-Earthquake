import functools
import inspect
import sys
import os
import warnings
import json
import numpy as np
import pandas as pd

from scipy import interpolate
from tqdm import tqdm


def window_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        window_size = kwargs['window_size']
        window_stride = kwargs['window_stride']
        verbose = True if "verbose" not in kwargs else kwargs['verbose']
        desc_line = get_function_descriptor(func, kwargs) if "desc_line" not in kwargs else kwargs['desc_line']

        assert self.data.shape[0] >= window_size, "Dataframe size is too small for a chosen window size"
        if not isinstance(verbose, bool):
            raise TypeError("verbose parameter must be a bool type!")
        if not isinstance(desc_line, str):
            raise TypeError("desc_line parameter must be a string type!")
        if not isinstance(window_size, int):
            raise TypeError("window_size parameter must be an integer type!")
        if not isinstance(window_stride, int):
            raise TypeError("window_stride parameter must be an integer type!")

        feature_name = self.update_name(desc_line)
        self.update_path()
        if self.exists():
            return self.read_feature()

        temp = []
        df = self.data
        if window_size is None:
            temp = func(self, df, *args, **kwargs).data
        else:
            for i in tqdm(range(0, df.shape[0] - window_size + window_stride, window_stride),
                          desc=desc_line, file=sys.stdout, disable=not verbose):
                batch = df.iloc[i: i + window_size]
                temp.append(func(self, batch, *args, **kwargs).data)

        if verbose:
            tqdm.write(f"\t window decorator for {func.__name__}: ")
            tqdm.write("\t - window size: {}".format(window_size))
            tqdm.write("\t - window stride: {}".format(window_stride))

        if hasattr(temp[0], 'shape') and temp[0].shape is not ():
            out_features = temp[0].shape[0]
            column_names = [feature_name + '_' + str(i) for i in range(out_features)]
        else:
            column_names = [feature_name]

        self.data = pd.DataFrame(temp, columns=column_names)
        return self
    return wrapper


def rolling_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        verbose = True if "verbose" not in kwargs else kwargs['verbose']
        desc_line = get_function_descriptor(func, kwargs) if "desc_line" not in kwargs else kwargs['desc_line']

        if not isinstance(verbose, bool):
            raise TypeError("verbose parameter must be a bool type!")
        if not isinstance(desc_line, str):
            raise TypeError("desc_line parameter must be a string type!")

        feature_name = self.update_name(desc_line)
        self.update_path()
        if self.exists():
            return self.read_feature()

        df = self.data
        temp = func(self, df, *args, **kwargs).data

        if verbose:
            tqdm.write(f"\t Rolling decorator for {func.__name__}: ")

        column_names = [feature_name]
        self.data = pd.DataFrame(temp, columns=column_names)
        return self

    return wrapper


def process_df(df, features, default_window_size, default_window_stride, df_path=None, is_test=True):

    """
    Data processing is done in three main steps:
    1) Calculate all features listed in configuration file (dp_config.json)
    2) Append labels
    3) Perform data resampling in case different window sizes are used

    df : raw data pandas DataFrame
    features : list of features specified in dp_config.json file
    default_window_size :
    default_window_stride :
    df_path : path to a folder where the processed pandas DataFrame will be stored

    Returns
    -------
    Processed pandas DataFrame
    """
    from feature import Feature

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
        data_processed, feature_name = calculate_feature(df.drop(['ttf'], axis=1), feature, *general_params)
        temp_data[feature_name] = data_processed

    if is_test:
        # append column with labels
        try:
            temp_data['ttf'] = Feature(df=df['ttf'], save_dir=df_path).w_last_elem(df['ttf'],
                                                                                   window_size=default_window_size,
                                                                                   window_stride=default_window_stride,
                                                                                   desc_line="ttf").data
        except KeyError as e:
            raise KeyError(f"Labels can't be calculated, key: {e} is missing")

    # perform resampling if needed
    resulted_size = temp_data['ttf'].shape[0]
    for k, v in temp_data.items():
        temp_data[k] = resample_columns(v, resulted_size)

    return pd.concat(temp_data.values(), axis=1)


def calculate_feature(df, feature, default_window_size, default_window_stride, save_dir):

    from feature import Feature

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

    from feature import Feature

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
