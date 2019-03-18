import numpy as np
import pandas as pd
import functools
import scipy.signal
import tsfresh
import inspect
from tqdm import tqdm
import json


def window_decorator(window_size=None):
    if window_size is None:
        with open("dp_config.json") as config:
            params = json.load(config)
        window_size = params['window_size']

    def window_calc(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            temp = []
            df = args[0]
            inspect_params = inspect.getfullargspec(func)

            if kwargs:
                desc_line = func.__name__ + "({}, ".format(*inspect_params.args) + ', '.join(
                    "{}={})".format(k, v) for k, v in kwargs.items())
            else:
                desc_line = func.__name__ + "({})".format(*inspect_params.args)

            for i in tqdm(range(0, df.shape[0], window_size),
                          desc=desc_line):
                batch = df.iloc[i: i + window_size].values
                temp.append(func(batch, *args, **kwargs))

            return pd.DataFrame(temp, columns={desc_line})
        return wrapper
    return window_calc


"""
Custom routines

"""


@window_decorator()
def w_psd(df, *args, fs=4e6, **kwargs):
    return np.sum(scipy.signal.periodogram(df, fs=fs)[1])


@window_decorator()
def w_labels(df, *args, **kwargs):
    return df[-1]


"""
numpy routines

"""


@window_decorator()
def w_min(df, *args, **kwargs):
    return np.min(df)


@window_decorator()
def w_max(df, *args, **kwargs):
    return np.max(df)


@window_decorator()
def w_mean(df, *args, **kwargs):
    return np.mean(df)


@window_decorator()
def w_std(df, *args, **kwargs):
    return np.std(df)


"""
tsfresh routines

"""


@window_decorator()
def w_abs_energy(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.abs_energy(df)


@window_decorator()
def w_absolute_sum_of_changes(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(df)


@window_decorator()
def w_autocorrelation(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.autocorrelation(df, lag=lag)


@window_decorator()
def w_binned_entropy(df, *args, max_bins=10, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.binned_entropy(df, max_bins=max_bins)


@window_decorator()
def w_c3(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.c3(df, lag=lag)


@window_decorator()
def w_count_above_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.count_above_mean(df)


@window_decorator()
def w_count_below_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.count_below_mean(df)


@window_decorator()
def w_first_location_of_maximum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.first_location_of_maximum(df)


@window_decorator()
def w_first_location_of_minimum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.first_location_of_minimum(df)


@window_decorator()
def w_kurtosis(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.kurtosis(df)


@window_decorator()
def w_large_standard_deviation(df, *args, r=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.large_standard_deviation(df, r=r)


@window_decorator()
def w_last_location_of_minimum(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(df)


@window_decorator()
def w_longest_strike_above_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(df)


@window_decorator()
def w_longest_strike_below_mean(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(df)


@window_decorator()
def w_mean_abs_change(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_abs_change(df)


@window_decorator()
def w_mean_change(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_change(df)


@window_decorator()
def w_mean_second_derivative_central(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(df)


@window_decorator()
def w_median(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.median(df)


@window_decorator()
def w_number_crossing_m(df, *args, m=0.1, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.number_crossing_m(df, m=m)


@window_decorator()
def w_number_cwt_peaks(df, *args, n=1, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(df, n=n)


@window_decorator()
def w_quantile(df, *args, q=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.quantile(df, q=q)


@window_decorator()
def w_ratio_beyond_r_sigma(df, *args, r=0.5, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.ratio_beyond_r_sigma(df, r=r)

# TODO: fix (freezes for some reason)
# @window_decorator()
# def w_sample_entropy(df, *args, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.sample_entropy(df)


@window_decorator()
def w_skewness(df, *args, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.skewness(df)

# TODO: fix param r (does not make sense)
# @window_decorator()
# def w_symmetry_looking(df, *args, r=0.1, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.symmetry_looking(df, r=r)


@window_decorator()
def w_skewness(df, *args, lag=100, **kwargs):
    return tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df, lag=lag)
