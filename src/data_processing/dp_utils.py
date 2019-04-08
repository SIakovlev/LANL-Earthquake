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

        # TODO: fix this hack
        if window_size >= df.shape[0]:
            # tqdm.write(f"{desc_line}:")
            # tqdm.write("\t window decorator: ")
            # tqdm.write("\t - window size: {}".format(window_size))
            temp = self.unwrapped(df.values, *args, **kwargs)
            return pd.DataFrame(temp, columns={desc_line})
        else:
            temp = []
            for i in tqdm(range(0, df.shape[0], window_size),
                          desc=desc_line, file=sys.stdout):
                batch = df.iloc[i: i + window_size].values
                temp.append(self.unwrapped(batch, *args, **kwargs))
            # tqdm.write("\t window decorator: ")
            # tqdm.write("\t - window size: {}".format(window_size))
            if hasattr(temp[0], 'shape') and temp[0].shape is not ():
                out_features = temp[0].shape[0]
                column_names = [desc_line + '_' + str(i) for i in range(out_features)]
            else:
                column_names = [desc_line]


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

        x_old = np.arange(0, old_size)
        f = interpolate.interp1d(x_old, df[col_name])
        x_new = np.linspace(0, old_size - 1, resulted_size)
        y_new = f(x_new)
        return pd.DataFrame(y_new, columns={col_name})

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

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.autocorrelation(df, lag=lag)


@DumpDecorator
@WindowDecorator
def w_binned_entropy(df, *args, max_bins=10, **kwargs):
    """

    Performs entropy based discretisation

    See description here:
    https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.binned_entropy

    Parameters
    ----------
    df : pandas DataFrame
    args :
    max_bins :
    kwargs :

    Returns
    -------

    """

    return tsfresh.feature_extraction.feature_calculators.binned_entropy(df, max_bins=max_bins)


@DumpDecorator
@WindowDecorator
def w_c3(df, *args, lag=100, **kwargs):
    """
    c3 measure was proposed in [1] as a measure of non linearity in the time series.

    For details see:
    https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.c3

    Parameters
    ----------
    df : pandas DataFrame
    args :
    lag : the lag in samples
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.c3(df, lag=lag)


@DumpDecorator
@WindowDecorator
def w_count_above_mean(df, *args, **kwargs):
    """
    Returns the number of values in x that are higher than the mean of x

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.count_above_mean(df)


@DumpDecorator
@WindowDecorator
def w_count_below_mean(df, *args, **kwargs):
    """
    Returns the number of values in x that are lower than the mean of x

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.count_below_mean(df)


@DumpDecorator
@WindowDecorator
def w_first_location_of_maximum(df, *args, **kwargs):
    """
    Returns the first location of the maximum value of x. The position is calculated relatively to the length of x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.first_location_of_maximum(df)


@DumpDecorator
@WindowDecorator
def w_first_location_of_minimum(df, *args, **kwargs):
    """
    Returns the first location of the minimal value of x. The position is calculated relatively to the length of x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.first_location_of_minimum(df)


@DumpDecorator
@WindowDecorator
def w_kurtosis(df, *args, **kwargs):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.kurtosis(df)


@DumpDecorator
@WindowDecorator
def w_large_standard_deviation(df, *args, r=0.5, **kwargs):
    """
    Boolean variable denoting if the standard dev of x is higher than ‘r’ times the range = difference between max and min of x.

    Hence it checks if

    std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    r : threshold value (see description)
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.large_standard_deviation(df, r=r)


@DumpDecorator
@WindowDecorator
def w_last_location_of_maximum(df, *args, **kwargs):
    """
    Returns the relative last location of the maximum value of x. The position is calculated relatively to the length of x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(df)


@DumpDecorator
@WindowDecorator
def w_last_location_of_minimum(df, *args, **kwargs):
    """
    Returns the relative last location of the minimum value of x. The position is calculated relatively to the length of x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(df)


@DumpDecorator
@WindowDecorator
def w_longest_strike_above_mean(df, *args, **kwargs):
    """
    Returns the length of the longest consecutive subsequence in x that is larger than the mean of x

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(df)


@DumpDecorator
@WindowDecorator
def w_longest_strike_below_mean(df, *args, **kwargs):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(df)


@DumpDecorator
@WindowDecorator
def w_mean_abs_change(df, *args, **kwargs):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    \frac{1}{n} \sum_{i=1,\ldots, n-1} | x_{i+1} - x_{i}|

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.mean_abs_change(df)


@DumpDecorator
@WindowDecorator
def w_mean_change(df, *args, **kwargs):
    """
    Returns the mean over the differences between subsequent time series values which is

    \frac{1}{n} \sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.mean_change(df)


@DumpDecorator
@WindowDecorator
def w_mean_second_derivative_central(df, *args, **kwargs):
    """
    Returns the mean value of a central approximation of the second derivative

    \frac{1}{n} \sum_{i=1,\ldots, n-1}  \frac{1}{2} (x_{i+2} - 2 \cdot x_{i+1} + x_i)

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(df)


@DumpDecorator
@WindowDecorator
def w_median(df, *args, **kwargs):
    """
    Returns the median of x

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """

    return tsfresh.feature_extraction.feature_calculators.median(df)


@DumpDecorator
@WindowDecorator
def w_number_crossing_m(df, *args, m=0.1, **kwargs):
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa.

    If you set m to zero, you will get the number of zero crossings.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    m : threshold value (see description)
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.number_crossing_m(df, m=m)


@DumpDecorator
@WindowDecorator
def w_number_cwt_peaks(df, *args, n=1, **kwargs):
    """
    This feature calculator searches for different peaks in x.

    To do so, x is smoothed by a ricker wavelet and for widths ranging from 1 to n.

    This feature calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    Parameters
    ----------
    df : pandas DataFrame
    args :
    n : max range value (see description)
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(df, n=n)


@DumpDecorator
@WindowDecorator
def w_quantile(df, *args, q=0.5, **kwargs):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    q (float) : quantile value (see description)
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.quantile(df, q=q)


@DumpDecorator
@WindowDecorator
def w_ratio_beyond_r_sigma(df, *args, r=0.5, **kwargs):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    r (float) : multiplier value (see description)
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.ratio_beyond_r_sigma(df, r=r)

# TODO: fix (freezes for some reason)
# @window_decorator()
# def w_sample_entropy(df, *args, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.sample_entropy(df)


@DumpDecorator
@WindowDecorator
def w_skewness(df, *args, **kwargs):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G1).

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.skewness(df)

# TODO: fix param r (does not make sense)
# @window_decorator()
# def w_symmetry_looking(df, *args, r=0.1, **kwargs):
#     return tsfresh.feature_extraction.feature_calculators.symmetry_looking(df, r=r)


@DumpDecorator
@WindowDecorator
def w_time_reversal_asymmetry_statistic(df, *args, lag=100, **kwargs):
    """
    Boolean variable denoting if the distribution of x looks symmetric. This is the case if

    | mean(X)-median(X)| < r * (max(X)-min(X))

    Parameters
    ----------
    df : pandas DataFrame
    args :
    lag : the lag in samples
    kwargs :

    Returns
    -------
    (bool)

    """
    return tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df, lag=lag)


"""
Other libraries

"""


@DumpDecorator
@WindowDecorator
def w_savgol_filter(df, *args, window_length=101, polyorder=1, **kwargs):
    if window_length > df.shape[0]:
        raise ValueError(f"Specified window length ({window_length}) is greater than dataframe size ({df.shape[0]})")
    return savgol_filter(df, window_length=window_length, polyorder=polyorder)


"""
Feets library

- API docs: https://feets.readthedocs.io/en/latest/api/feets.html


"""


@DumpDecorator
@WindowDecorator
def w_con(df, *args, **kwargs):
    """
    To calculate Con, we count the number of three consecutive data points that are brighter or fainter than 2σ
    and normalize the number by N−2.

    For a normal distribution and by considering just one star, Con should take values close to 0.045

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Con'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_eta_e(df, *args, **kwargs):
    """
    Variability index η is the ratio of the mean of the square of successive differences to the variance of data points.

    The index was originally proposed to check whether the successive data points are independent or not.
    In other words, the index was developed to check if any trends exist in the data (von Neumann 1941

    Link: https://feets.readthedocs.io/en/latest/api/feets.extractors.html#module-feets.extractors.ext_eta_e

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------
    Variability index η
    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Eta_e'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_gskew(df, *args, **kwargs):
    """
    Median-of-magnitudes based measure of the skew.

    Formula is given here:
    https://feets.readthedocs.io/en/latest/api/feets.extractors.html#module-feets.extractors.ext_gskew

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Gskew'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_linear_trend(df, *args, **kwargs):
    """
    Slope of a linear fit to a signal.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['LinearTrend'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_median_BRP(df, *args, **kwargs):
    """
    MedianBRP (Median buffer range percentage)

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------
    Fraction (<= 1) of points within amplitude/10 of the median magnitude
    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['MedianBRP'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_pair_slope_trend(df, *args, **kwargs):
    """
    Considering the last 30 (time-sorted) measurements of a signal magnitude,
    the fraction of increasing first differences minus the fraction of decreasing first differences.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['PairSlopeTrend'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_q31(df, *args, **kwargs):
    """
    Q3−1 is the difference between the third quartile, Q3, and the first quartile, Q1, of a signal.

    Q1 is a split between the lowest 25% and the highest 75% of data.
    Q3 is a split between the lowest 75% and the highest 25% of data.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Q31'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_slottedA_length(df, *args, **kwargs):
    """
    In slotted autocorrelation, time lags are defined as intervals or slots instead of single values.
    The slotted autocorrelation function at a certain time lag slot is computed by averaging the cross product between
    samples whose time differences fall in the given slot.

    TODO: add this
    There is a parameter T: slot size in days (wtf???) (default=1).

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0,size-1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['SlottedA_length'], data=['time','magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_percent_difference_flux_percentile(df, *args, **kwargs):
    """
    Ratio of F5,95 over the median magnitude.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['PercentDifferenceFluxPercentile'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]


@DumpDecorator
@WindowDecorator
def w_flux_percentile_ratio_mid80(df, *args, **kwargs):
    """
    flux_percentile_ratio_mid80: ratio F10,90/F5,95

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['FluxPercentileRatioMid80'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_flux_percentile_ratio_mid50(df, *args, **kwargs):
    """
    flux_percentile_ratio_mid50: ratio F25,75/F5,95

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['FluxPercentileRatioMid50'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_flux_percentile_ratio_mid20(df, *args, **kwargs):
    """
    flux_percentile_ratio_mid50: ratio F40,60/F5,95

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['FluxPercentileRatioMid20'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_flux_percentile_ratio_sum(df, *args, **kwargs):
    """
    Sum of flux_percentile_ratio_mid20+flux_percentile_ratio_mid50+flux_percentile_ratio_mid80

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['FluxPercentileRatioMid20', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid80'], data=['time', 'magnitude'])
    return np.sum(fs.extract(*t_m)[1])




@DumpDecorator
@WindowDecorator
def w_freq2_harmonics_rel_phase_2(df, *args, **kwargs):
    """
    Freq2_harmonics_rel_phase_2

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Freq2_harmonics_rel_phase_2'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_max_slope(df, *args, **kwargs):
    """
    Maximum absolute magnitude slope between two consecutive observations.

    Examining successive (time-sorted) magnitudes, the maximal first difference (value of delta magnitude over delta time)

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['MaxSlope'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_rcs(df, *args, **kwargs):
    """
    Rcs - Range of cumulative sum (Rcs)

    Rcs is the range of a cumulative sum (Ellaway 1978) of each light-curve and is defined as:
    Rcs=max(S)−min(S)

    S=1/Nσ ∑(mi−mean(m))

    where max(min) is the maximum (minimum) value of S and l=1,2,…,N.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['Rcs'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]

@DumpDecorator
@WindowDecorator
def w_percent_amplitude(df, *args, **kwargs):
    """
    Largest percentage difference between either the max or min magnitude and the median.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    size = df.shape[0]
    time = np.linspace(0, size - 1, size)
    magnitude = df
    t_m = [time, magnitude]
    fs = feets.FeatureSpace(only=['PercentAmplitude'], data=['time', 'magnitude'])
    return fs.extract(*t_m)[1][0]




