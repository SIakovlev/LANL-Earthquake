import numpy as np
import pandas as pd
import functools
import scipy.signal
from scipy import interpolate
import tsfresh
import inspect
import sys
import json
import os
from tqdm import tqdm
from scipy.signal import savgol_filter


class DumpDecorator:
    def __init__(self, func):
        self._func = func

    def __call__(self, target, *args, **kwargs):
        df = self._func(target, *args, **kwargs)
        if "save_path" in kwargs:
            tqdm.write("\t dump decorator: ")
            tqdm.write("\t - save_path: {}".format(kwargs["save_path"]))
            df.to_hdf(kwargs["save_path"], key='table')
        return df


class WindowDecorator:
    def __init__(self, f):
        self.unwrapped = f
        functools.update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        if self.unwrapped is None:
            self.unwrapped = args[0]

        df = args[0]
        window_size = kwargs['window_size']
        desc_line = kwargs['desc_line']

        if window_size >= df.shape[0]:
            tqdm.write(f"{desc_line}:")
            tqdm.write("\t window decorator: ")
            tqdm.write("\t - window size: {}".format(window_size))
            return pd.DataFrame(df.values, columns={desc_line})
        else:
            temp = []
            for i in tqdm(range(0, df.shape[0], window_size),
                          desc=desc_line, file=sys.stdout):
                batch = df.iloc[i: i + window_size].values
                temp.append(self.unwrapped(batch, *args, **kwargs))
            tqdm.write("\t window decorator: ")
            tqdm.write("\t - window size: {}".format(window_size))

            return pd.DataFrame(temp, columns={desc_line})


def process_df(df, routines, default_window_size):
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
    this_module_name = sys.modules[__name__]
    temp_data = {}

    # calc all the features
    for routine in routines:
        if not routine['on']:
            continue
        func = getattr(this_module_name, routine["name"])
        func_params = routine['params']
        window_size = default_window_size if 'window_size' not in routine else routine['window_size']
        try:
            data = df[routine['column_name']] if routine['column_name'] in df.columns \
                else temp_data[routine['column_name']].squeeze()
        except KeyError as e:
            raise KeyError(f"Check your feature calculation order, key: {e} is missing")

        # TODO: add chaining to column names
        if func_params:
            desc_line = f"{func.__name__}(df, window_size={window_size}, " + \
                        ', '.join("{!s}={!r}".format(key, val) for (key, val) in func_params.items()) + ')'
        else:
            desc_line = f"{func.__name__}(df, window_size={window_size})"

        data_processed = func(data, window_size=window_size, desc_line=desc_line, **func_params)
        new_col_name = data_processed.columns.values.tolist()[0]
        temp_data[new_col_name] = data_processed

    # append column with labels
    try:
        temp_data['ttf'] = w_last_elem(df['ttf'], window_size=default_window_size, desc_line="ttf")
    except KeyError as e:
        raise KeyError(f"Labels can't be calculated, key: {e} is missing")

    # perform resampling if needed
    # TODO: do proper sampling
    resulted_size = temp_data['ttf'].shape[0]
    for k, v in temp_data.items():
        temp_data[k] = resample_column(v, resulted_size)

    res = pd.concat(temp_data.values(), axis=1)
    return res


def resample_column(df, resulted_size):
    """
    Perform resampling of dataframe df to the resulted size

    :param df:
    :type df:
    :param resulted_size:
    :type resulted_size:
    :return:
    :rtype:
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


"""
Custom routines

"""


# @DumpDecorator
@WindowDecorator
def w_psd(df, *args, fs=4e6, **kwargs):
    return np.sum(scipy.signal.periodogram(df, fs=fs)[1])


# @DumpDecorator
@WindowDecorator
def w_last_elem(df, *args, **kwargs):
    return df[-1]


"""
numpy routines

"""


@WindowDecorator
def w_min(df, *args, **kwargs):
    return np.min(df)


@WindowDecorator
def w_max(df, *args, **kwargs):
    return np.max(df)


@WindowDecorator
def w_mean(df, *args, **kwargs):
    return np.mean(df)


@WindowDecorator
def w_std(df, *args, **kwargs):
    return np.std(df)


"""
tsfresh routines

"""


@WindowDecorator
def w_abs_energy(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.abs_energy(df)


@WindowDecorator
def w_absolute_sum_of_changes(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(df)


@WindowDecorator
def w_autocorrelation(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.autocorrelation(df, lag=lag)


@WindowDecorator
def w_binned_entropy(df, *args, max_bins=10, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.binned_entropy(df, max_bins=max_bins)


@WindowDecorator
def w_c3(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.c3(df, lag=lag)


@WindowDecorator
def w_count_above_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.count_above_mean(df)


@WindowDecorator
def w_count_below_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.count_below_mean(df)


@WindowDecorator
def w_first_location_of_maximum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.first_location_of_maximum(df)


@WindowDecorator
def w_first_location_of_minimum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.first_location_of_minimum(df)


@WindowDecorator
def w_kurtosis(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.kurtosis(df)


@WindowDecorator
def w_large_standard_deviation(df, *args, r=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.large_standard_deviation(df, r=r)


@WindowDecorator
def w_last_location_of_minimum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(df)


@WindowDecorator
def w_longest_strike_above_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(df)


@WindowDecorator
def w_longest_strike_below_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(df)


@WindowDecorator
def w_mean_abs_change(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_abs_change(df)


@WindowDecorator
def w_mean_change(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_change(df)


@WindowDecorator
def w_mean_second_derivative_central(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(df)


@WindowDecorator
def w_median(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.median(df)


@WindowDecorator
def w_number_crossing_m(df, *args, m=0.1, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.number_crossing_m(df, m=m)


@WindowDecorator
def w_number_cwt_peaks(df, *args, n=1, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(df, n=n)


@WindowDecorator
def w_quantile(df, *args, q=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.quantile(df, q=q)


@WindowDecorator
def w_ratio_beyond_r_sigma(df, *args, r=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.ratio_beyond_r_sigma(df, r=r)

# TODO: fix (freezes for some reason)
# @window_decorator()
# def w_sample_entropy(df, *args, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.sample_entropy(df)


@WindowDecorator
def w_skewness(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.skewness(df)

# TODO: fix param r (does not make sense)
# @window_decorator()
# def w_symmetry_looking(df, *args, r=0.1, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.symmetry_looking(df, r=r)


@WindowDecorator
def w_skewness(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df, lag=lag)


"""
Other libraries

"""


@WindowDecorator
def w_savgol_filter(df, *args, window_length=101, polyorder=1, **kwargs):
    return savgol_filter(df, window_length=window_length, polyorder=polyorder)
