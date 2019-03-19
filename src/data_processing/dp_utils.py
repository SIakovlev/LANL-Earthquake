import numpy as np
import pandas as pd
import functools
import scipy.signal
import tsfresh
import inspect
import sys
import json
import os
from tqdm import tqdm
from scipy.signal import savgol_filter


class WindowDecorator:
    def __init__(self, f):
        self.f = f
        functools.update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        if self.f is None:
            self.f = args[0]

        temp = []
        df = args[0]
        window_size = kwargs['window_size']
        inspect_params = inspect.getfullargspec(self.f)

        if kwargs:
            desc_line = self.f.__name__ + "({}, ".format(*inspect_params.args) + ', '.join(
                "{}={})".format(k, v) for k, v in kwargs.items())
        else:
            desc_line = self.f.__name__ + "({})".format(*inspect_params.args)

        for i in tqdm(range(0, df.shape[0], window_size),
                      desc=desc_line, file=sys.stdout):
            batch = df.iloc[i: i + window_size].values
            temp.append(self.f(batch, *args, **kwargs))
        tqdm.write("\t window decorator: ")
        tqdm.write("\t - window size: {}".format(window_size))
        return pd.DataFrame(temp, columns={desc_line})


# TODO: remove soon if not used
def function_decorator(f, params):
    def filter_calc(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df = func(args[0], *args, **kwargs)
            temp = f(df.squeeze(), **params)
            tqdm.write("\t function decorator: ")
            tqdm.write("\t - {}{}".format(f.__name__, inspect.signature(f)))
            tqdm.write("\t - params: {}".format(params))
            return pd.DataFrame(temp, columns={func.__name__})
        return wrapper
    return filter_calc


def process_df(df, routines, default_window_size):
    this_module_name = sys.modules[__name__]
    temp_data = {}

    # calc all the features
    for routine in routines:
        if not routine['on']:
            continue
5        func = getattr(this_module_name, routine["name"])
        func_params = routine['params']
        window_size = default_window_size if 'window_size' not in routine else routine['window_size']
        try:
            data = df[routine['column_name']]
        except KeyError as e:
            raise KeyError(f"Check your feature calculation order, key: {e} is missing")

        data_processed = func(data, window_size=window_size, **func_params)
        new_col_name = data_processed.columns.values.tolist()[0]
        temp_data[new_col_name] = data_processed

    # perform scaling if needed
    # TODO: do scaling

    res = pd.concat(temp_data.values(), axis=1)
    return res


"""
TODO:
1) remove DataProcessing.py (done)
2) Add class wrapper for decorators (done)
3) Add chaining with a single column
4) Add variable window size support (with upsampling and downsampling + calculate window size) 

"""


"""
Custom routines

"""


@WindowDecorator
def w_psd(df, *args, fs=4e6, **kwargs):
    return np.sum(scipy.signal.periodogram(df, fs=fs)[1])


# TODO: not quite clear if this is easy to use
# @function_decorator(savgol_filter, {"window_length": 11, "polyorder": 1})
# @window_decorator()
# def wf_psd(df, *args, fs=4e6, **kwargs):
#     return np.sum(scipy.signal.periodogram(df, fs=fs)[1])


@WindowDecorator
def w_labels(df, *args, **kwargs):
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

