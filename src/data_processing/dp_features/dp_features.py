import numpy as np
import scipy.signal
from scipy.signal import savgol_filter
import tsfresh
import feets
from dp_utils import DumpDecorator, WindowDecorator, DataFrameDecorator
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=feets.ExtractorWarning)


"""
Custom routines

"""


@DumpDecorator
@WindowDecorator
def w_psd(df, *args, fs=4e6, **kwargs):
    """
    Calculates total power spectrum density of the dataframe and sums it up

    Parameters
    ----------
    df : pandas DataFrame
    args : None
    fs : sampling frequency
    kwargs :

    Returns
    -------
    A sum of spectral components of the dataframe
    """
    return np.sum(scipy.signal.periodogram(df, fs=fs)[1])


@DumpDecorator
@WindowDecorator
def w_periodogram(df, *args, fs=4e6, N=2000, **kwargs):
    """
    Calculates power spectrum density of the dataframe

    Parameters
    ----------
    df : pandas DataFrame
    args : None
    fs : sampling frequency
    kwargs :

    Returns
    -------
    Spectral components of the dataframe
    """
    # TODO: remove NaNs
    return scipy.signal.periodogram(df, fs=fs)[1][:N]


@DumpDecorator
@WindowDecorator
def w_spectrogramm_downsampled(df, *args, fs=4e6, nperseg=50000, noverlap=20000, mode='psd', **kwargs):
    """
    custom spectrogramm with downsampling afterwards in freq domain
    Parameters
    ----------
    df : pandas DataFrame
    args : None
    fs : sampling frequency
    kwargs :

    Returns
    -------

    """
    f, t, Sxx = scipy.signal.spectrogram(df, fs, nperseg=nperseg, noverlap=noverlap, mode=mode)
    smoothen = scipy.signal.convolve2d(Sxx, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
    smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
    smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
    return smoothen.T.flatten()


@DumpDecorator
@WindowDecorator
def w_last_elem(df, *args, **kwargs):
    """
    Get the last element of the dataframe

    Parameters
    ----------
    df : pandas DataFrame
    args : None
    kwargs : None

    Returns
    -------
    The last element of the dataframe
    """
    return df[-1]


@DumpDecorator
@DataFrameDecorator
def df_savgol_filter(df, *args, window_length=101, polyorder=1, **kwargs):
    """
    Apply a Savitzky-Golay 1d filter to a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
    args :
    window_length : The length of the filter window (i.e. the number of coefficients), must be a positive odd integer.
    polyorder : The order of the polynomial used to fit the samples. polyorder must be less than window_length
    kwargs :

    Returns
    -------

    """
    if not window_length % 2:
        raise ValueError(f"The length of the filter window (i.e. the number of coefficients), "
                         f"must be a positive odd integer. Your value: {window_length}")
    if window_length > df.shape[0]:
        raise ValueError(f"Specified window length ({window_length}) is greater than dataframe size ({df.shape[0]})")
    return savgol_filter(df, window_length=window_length, polyorder=polyorder)


@DumpDecorator
@DataFrameDecorator
def df_sta_lta(df, *args, sta_window=100, lta_window=1000, **kwargs):
    """
    Calculates STA/LTA ratio over windows sta_window and lta_window

    Parameters
    ----------
    df : pandas DataFrame
    args :
    sta_window : short-time average window
    lta_window : long-time average window
    kwargs :

    Returns
    -------

    """

    if sta_window > lta_window:
        raise ValueError(f"Short-time window can't be longer than long-time window!")

    return classic_sta_lta(df, sta_window, lta_window)


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta


def df_roll_std(df, *args, window_length=100, **kwargs):
    """

    Parameters
    ----------
    df :
    args :
    sta_window :
    lta_window :
    kwargs :

    Returns
    -------

    """
    return pd.DataFrame(df.rolling(window_length, min_periods=1).std(ddof=0))


"""
numpy routines

"""


@DumpDecorator
@WindowDecorator
def w_min(df, *args, **kwargs):
    return np.min(df)


@DumpDecorator
@WindowDecorator
def w_max(df, *args, **kwargs):
    return np.max(df)


@DumpDecorator
@WindowDecorator
def w_mean(df, *args, **kwargs):
    return np.mean(df)


@DumpDecorator
@WindowDecorator
def w_std(df, *args, **kwargs):
    return np.std(df)


"""
tsfresh routines

- Docs: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html

"""


@DumpDecorator
@WindowDecorator
def w_abs_energy(df, *args, **kwargs):
    """
    Calculates the absolute energy of the time series which is the sum over the squared values

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------
    The sum over the squared values

    """
    return tsfresh.feature_extraction.feature_calculators.abs_energy(df)


@DumpDecorator
@WindowDecorator
def w_absolute_sum_of_changes(df, *args, **kwargs):
    """
    Returns the sum over the absolute value of consecutive changes in the series

    Parameters
    ----------
    df : pandas DataFrame
    args :
    kwargs :

    Returns
    -------

    """
    return tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(df)


@DumpDecorator
@WindowDecorator
def w_approximate_entropy(df, *args, m=3, r=3, **kwargs):
    """
    Implements a vectorized Approximate entropy algorithm.

    https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters, but should be stable for N > 2000

    Parameters
    ----------
    df : pandas DataFrame
    args :
    m : (int) – Length of compared run of data
    r : (float) – Filtering level, must be positive
    kwargs :

    Returns
    -------
    Approximate entropy of the dataframe
    """
    return tsfresh.feature_extraction.feature_calculators.approximate_entropy(df, m, r)


@DumpDecorator
@WindowDecorator
def w_autocorrelation(df, *args, lag=100, **kwargs):
    """
    Calculates the autocorrelation of the specified lag

    Parameters
    ----------
    df : pandas DataFrame
    args :
    lag : the lag in samples
    kwargs :

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
