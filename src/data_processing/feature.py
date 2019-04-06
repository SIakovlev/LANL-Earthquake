import numpy as np
import scipy.signal
from scipy.signal import savgol_filter
import tsfresh
import feets
from dp_utils import window_decorator
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=feets.ExtractorWarning)


class Feature:
    def __init__(self, df, save_dir):
        self.data = df
        self.name = []
        self.save_dir = save_dir

    def dump(self, save_dir=None, ext='.h5'):
        save_path = self.save_dir if save_dir is None else save_dir
        save_path += '-'.join(self.name) + ext
        if save_path in os.listdir(save_dir):
            return self
        tqdm.write(f"\t - saving to: {save_path}")
        self.data.to_hdf(save_path, key='table')
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
        The last element of the dataframe
        """
        self.data = self.data[-1] if df in None else df[-1]
        return self

    @window_decorator
    def w_mean(self, df=None, *args, **kwargs):
        self.data = np.mean(self.data, axis=0) if df is None else np.mean(df, axis=0)
        return self

    @window_decorator
    def w_std(self, df=None, *args, **kwargs):
        self.data = np.std(self.data, axis=0) if df is None else np.std(df, axis=0)
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

        self.data = scipy.signal.periodogram(self.data, fs=fs)[1][:N] if df is None else \
        scipy.signal.periodogram(df, fs=fs)[1][:N]
        return self

    @window_decorator
    def w_spectrogramm_downsampled(self, df=None, *args, fs=4e6, nperseg=100, noverlap=20, mode='psd', **kwargs):
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
        f, t, Sxx = scipy.signal.spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap, mode=mode)
        smoothen = scipy.signal.convolve2d(Sxx, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        smoothen = scipy.signal.convolve2d(smoothen, np.array([[0.25, 0.25, 0.25, 0.25]]).T, mode='full')[::4]
        self.data = smoothen.T.flatten()
        return self
