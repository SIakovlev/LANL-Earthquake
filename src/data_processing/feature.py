
import scipy.signal
import tsfresh
import feets
import warnings
import os
import functools
import sys
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore", category=feets.ExtractorWarning)


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


class Feature:
    def __init__(self, df, save_dir, ext='.h5'):
        self.data = df
        self.save_dir = save_dir

        self.__name = []
        self.__path = ''
        self.__ext = ext

    def update_name(self, name):
        if not isinstance(name, str):
            raise TypeError("name should be string type")
        self.__name.append(name)
        return '-'.join(self.__name)

    def get_name(self):
        return '-'.join(self.__name)

    def update_path(self, save_dir=None):
        self.__path = self.save_dir if save_dir is None else save_dir
        self.__path += self.get_name() + self.__ext

    def get_path(self):
        return self.__path

    def exists(self):
        return self.get_name() + self.__ext in os.listdir(self.save_dir)

    def dump(self, save_dir=None):
        self.update_path(save_dir)
        if self.exists():
            return self
        else:
            tqdm.write(f"\t - saving to: {self.__path}")
            self.data.to_hdf(self.__path, key='table')
            return self

    def read_feature(self):
        try:
            self.data = pd.read_hdf(self.__path, key='table')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No file: {e}")
        return self

    """
    Custom methods

    """

    @window_decorator
    def w_psd_sum(self, df=None, *args, fs=4e6, **kwargs):
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
        data = self.data if df is None else df
        self.data = np.sum(scipy.signal.periodogram(data.values.squeeze(), fs=fs)[1])
        return self

    @window_decorator
    def w_psd(self, df=None, *args, fs=4e6, **kwargs):
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
        spectral components of the dataframe
        """
        data = self.data if df is None else df
        self.data = scipy.signal.periodogram(data.values.reshape(-1,), fs=fs)[1]
        return self

    @window_decorator
    def w_last_elem(self, df=None, *args, **kwargs):
        """
        Get the last element of the dataframe

        Parameters
        ----------
        df : pandas DataFrame
        args : None
        kwargs : None

        Returns
        -------
        """
        data = self.data if df is None else df
        self.data = data.values[-1]
        return self

    @window_decorator
    def w_periodogram(self, df=None, *args, fs=4e6, N=2000, **kwargs):
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
        data = self.data if df is None else df
        self.data = scipy.signal.periodogram(data.values.squeeze(), fs=fs)[1][:N]
        return self

    @window_decorator
    def w_spectrogram_downsampled(self, df=None, *args, fs=4e6, nperseg=100, noverlap=20, mode='psd', **kwargs):
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
        data = self.data if df is None else df
        f, t, Sxx = scipy.signal.spectrogram(data.values.squeeze(), fs, nperseg=nperseg, noverlap=noverlap, mode=mode)
        smoothen = scipy.signal.convolve2d(Sxx, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        self.data = smoothen.T.flatten()
        return self

    @rolling_decorator
    def r_savgol_filter(self, df=None, *args, window_length=101, polyorder=1, **kwargs):
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
        data = self.data if df is None else df
        if not window_length % 2:
            raise ValueError(f"The length of the filter window (i.e. the number of coefficients), "
                             f"must be a positive odd integer. Your value: {window_length}")
        if window_length > data.shape[0]:
            raise ValueError(
                f"Specified window length ({window_length}) is greater than the dataframe size ({data.shape[0]})")
        self.data = savgol_filter(data.squeeze(), window_length=window_length, polyorder=polyorder)
        return self

    @rolling_decorator
    def r_sta_lta(self, df=None, *args, sta_window=1000, lta_window=10000, **kwargs):
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

        data = self.data if df is None else df
        if sta_window > lta_window:
            raise ValueError(f"Short-time window can't be longer than long-time window!")
        self.data = classic_sta_lta(data, sta_window, lta_window)
        return self

    """
    Numpy based methods

    """

    @window_decorator
    def w_av_change_abs(self, df=None, *args, axis=0, **kwargs):
        data = self.data if df is None else df
        self.data = np.mean(np.diff(data.values.squeeze()), axis=axis)
        return self

    @window_decorator
    def w_av_change_rate(self, df=None, *args, axis=0, **kwargs):
        data = self.data if df is None else df
        self.data = np.mean(np.nonzero((np.diff(data.values.squeeze()) /data.values.squeeze()[:-1]))[0],axis=axis)
        return self

    @window_decorator
    def w_min(self, df=None, *args, axis=0, **kwargs):
        data = self.data if df is None else df
        self.data = np.min(data.values, axis=axis)
        return self


    @window_decorator
    def w_mean(self, df=None, *args, axis=0, **kwargs):
        data = self.data if df is None else df
        self.data = np.mean(data.values, axis=axis)
        return self

    @window_decorator
    def w_std(self, df=None, *args, axis=0, **kwargs):
        data = self.data if df is None else df
        self.data = np.std(data.values, axis=axis)
        return self

    @window_decorator
    def w_max(self, df=None, *args, **kwargs):
        data = self.data if df is None else df
        self.data = np.max(data.values)
        return self

    @window_decorator
    def w_min(self, df=None, *args, **kwargs):
        data = self.data if df is None else df
        self.data = np.min(data.values)
        return self
    
    """
    tsfresh based methods

    """

    @window_decorator
    def w_abs_energy(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.abs_energy(data.values.squeeze())
        return self

    @window_decorator
    def w_absolute_sum_of_changes(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(data.values.squeeze())
        return self

    @window_decorator
    def w_approximate_entropy(self, df=None, *args, m=3, r=3, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.approximate_entropy(data.values.squeeze(),
                                                                                       m=m,
                                                                                       r=r)
        return self

    def w_fft_rmean_last_5000(self, df=None, *args, lag=100, **kwargs):
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
        data = self.data if df is None else df
        zc = np.fft.fft(data.values.squeeze())
        self.data = np.real(zc[-5000:]).mean()
        return self

    @window_decorator
    def w_irq(self, df=None, *args, lag=100, **kwargs):
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
        data = self.data if df is None else df
        self.data = np.subtract(*np.percentile(data.values.squeeze(), [75, 25]))
        return self


    @window_decorator
    def w_fft_lmean_last_5000(self, df=None, *args, lag=100, **kwargs):
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
        data = self.data if df is None else df
        zc = np.fft.fft(data.values.squeeze())
        self.data = np.imag(zc[-5000:]).mean()
        return self

    @window_decorator
    def w_autocorrelation(self, df=None, *args, lag=10, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.autocorrelation(data.values.squeeze(),
                                                                                   lag=lag)
        return self

    @window_decorator
    def w_binned_entropy(self, df=None, *args, max_bins=100, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.binned_entropy(data.values.squeeze(),
                                                                                  max_bins=max_bins)
        return self

    @window_decorator
    def w_c3(self, df=None, *args, lag=100, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.c3(data.values.squeeze(),
                                                                      lag=lag)
        return self

    @window_decorator
    def w_count_above_mean(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.count_above_mean(data.values.squeeze())
        return self

    @window_decorator
    def w_count_below_mean(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.count_below_mean(data)
        return self
    
    @window_decorator
    def w_first_location_of_maximum(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.first_location_of_maximum(data.values.squeeze())
        return self
    
    @window_decorator
    def w_first_location_of_minimum(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.first_location_of_minimum(data.values.squeeze())
        return self
    
    @window_decorator
    def w_kurtosis(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.kurtosis(data.values.squeeze())
        return self

    @window_decorator
    def w_large_standard_deviation(self, df=None, *args, r=0.5, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.large_standard_deviation(data.values.squeeze(), 
                                                                                            r=r)
        return self

    @window_decorator
    def w_last_location_of_maximum(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.last_location_of_maximum(data.values.squeeze())
        return self

    @window_decorator
    def w_last_location_of_minimum(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.last_location_of_minimum(data.values.squeeze())
        return self

    @window_decorator
    def w_longest_strike_above_mean(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(data.values.squeeze())
        # del data
        # gc.collect()
        return self

    
    @window_decorator
    def w_longest_strike_below_mean(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(data.values.squeeze())
        # del data
        # gc.collect()
        return self

    @window_decorator
    def w_mean_abs_change(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.mean_abs_change(data.values.squeeze())
        return self

    @window_decorator
    def w_mean_change(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.mean_change(data.values.squeeze())
        return self

    @window_decorator
    def w_mean_second_derivative_central(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(data.values.squeeze())
        return self

    @window_decorator
    def w_median(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = np.median(data.values.squeeze())
        # self.data = tsfresh.feature_extraction.feature_calculators.median(data.values.squeeze())
        return self

    @window_decorator
    def w_number_crossing_m(self, df=None, *args, m=0.1, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.number_crossing_m(data.values.squeeze(), m=m)
        return self

    @window_decorator
    def w_number_cwt_peaks(self, df=None, *args, n=1, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.number_cwt_peaks(data.values.squeeze(), n=n)
        return self

    @window_decorator
    def w_quantile(self, df=None, *args, q=0.05, **kwargs):
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
        data = self.data if df is None else df
        self.data = np.quantile(data.values.squeeze(), q=q)
        # self.data = tsfresh.feature_extraction.feature_calculators.quantile(data.values.squeeze(), q=q)
        return self
    
    @window_decorator
    def w_ratio_beyond_r_sigma(self, df=None, *args, r=2, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.ratio_beyond_r_sigma(data.values.squeeze(), r=r)
        return self

    # TODO: fix (freezes for some reason)
    # @window_decorator
    # def w_sample_entropy(self, df=None, *args, **kwargs):
    #     return tsfresh.feature_extraction.feature_calculators.sample_entropy(df)

    @window_decorator
    def w_skewness(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.skewness(data.values.squeeze())
        return self

    # TODO: fix param r (does not make sense)
    # @window_decorator
    # def w_symmetry_looking(self, df=None, *args, r=0.1, **kwargs):
    #     pass

    @window_decorator
    def w_time_reversal_asymmetry_statistic(self, df=None, *args, lag=100, **kwargs):
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
        data = self.data if df is None else df
        self.data = tsfresh.feature_extraction.feature_calculators.\
            time_reversal_asymmetry_statistic(data.values.squeeze(), lag=lag)
        return self

    """
    Feets library

    - API docs: https://feets.readthedocs.io/en/latest/api/feets.html


    """

    @window_decorator
    def w_con(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = data.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['Con'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_percent_difference_flux_percentile(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = data.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['PercentDifferenceFluxPercentile'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_eta_e(self, df=None, *args, **kwargs):
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

        """
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['Eta_e'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self
    
    @window_decorator
    def w_gskew(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['Gskew'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_linear_trend(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['LinearTrend'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_median_BRP(self, df=None, *args, **kwargs):
        """
        MedianBRP (Median buffer range percentage). Calculates fraction (<= 1) of points
        within amplitude/10 of the median magnitude

        Parameters
        ----------
        df : pandas DataFrame
        args :
        kwargs :

        Returns
        -------

        """
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['MedianBRP'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_pair_slope_trend(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['PairSlopeTrend'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_q31(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['Q31'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    @window_decorator
    def w_flux_percentile_ratio_sum(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(
            only=['FluxPercentileRatioMid20', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid80'],
            data=['time', 'magnitude'])
        self.data = np.sum(fs.extract(*t_m)[1])
        return self

    @window_decorator
    def w_rcs(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['Rcs'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self



    @window_decorator
    def w_slottedA_length(self, df=None, *args, **kwargs):
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
        data = self.data if df is None else df
        size = df.shape[0]
        time = np.linspace(0, size - 1, size)
        magnitude = data.values.squeeze()
        t_m = [time, magnitude]
        fs = feets.FeatureSpace(only=['SlottedA_length'], data=['time', 'magnitude'])
        self.data = fs.extract(*t_m)[1][0]
        return self

    """
    Pandas based methods

    """

    @rolling_decorator
    def r_std(self, df=None, window_size=100, *args, **kwargs):
        """

        Parameters
        ----------
        df : pandas DataFrame
        args :
        kwargs :

        Returns
        -------

        """
        data = self.data if df is None else df
        self.data = data.rolling(window_size, min_periods=1).std(ddof=0).values
        return self

    @rolling_decorator
    def r_mean(self, df=None, window_size=100, *args, **kwargs):
        """

        Parameters
        ----------
        df : pandas DataFrame
        args :
        kwargs :

        Returns
        -------

        """
        data = self.data if df is None else df
        self.data = data.rolling(window_size, min_periods=1).mean().values
        return self
