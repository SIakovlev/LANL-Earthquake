import pandas as pd
import numpy as np

"""
DataProcessing module.
Date: 16.02.2019
Description:
"""





class DataProcessorBase:
    """
    Base class for all the data processing
    """
    def __init__(self, column_names=None, suffix=None, **kwargs):
        """
        Create instance of data processing object that will process specified columns of the Dataframe
        :param column_names: list of strings
        :param suffix: str
        :param kwargs:
        """
        self.column_names = column_names
        self.suffix = suffix

    def load(self, path, key='table', **kwargs):
        """
        Probably should remove load/save functionality from dp and put it in the Dataframe wrapper
        :param path:
        :param key:
        :param kwargs:
        :return:
        """
        # TODO: fix for large frames: (generators?)

        return pd.read_hdf(path, key)

    def save(self, obj, path, key='table', **kwargs):
        """
            Probably should remove load/save functionality from dp and put it in the Dataframe wrapper
            :param path:
            :param key:
            :param kwargs:
            :return:
        """

        # TODO: fix for large frames: (generators?)

        obj.to_hdf(path, key)

    def data_loader(self, path, **kargs):
        """
        Remove soon
        :param path:
        :param kargs:
        :return:
        """
        return pd.read_csv(path, dtype={'s': np.int16, 'ttf': np.float32})

    def __call__(self, df, *args, **kwargs):
        """
        Perform processing on the dataframe
        :param df: Pandas.DataFrame
        :param args:
        :param kwargs:
        :return: modified df: Pandas.Dataframe
        """
        raise NotImplementedError
        return df


class DataProcessorMin(DataProcessorBase):
    """
    Calculate running min
    """
    def __init__(self, window_length=None, **kwargs):
        super(DataProcessorMin, self).__init__(**kwargs)
        if type(window_length) is not int:
            raise AttributeError("Window length has to be int")
        self.window = window_length
        if not self.suffix:
            self.suffix = '_min_' + str(self.window)

    def __call__(self, df, *args, **kwargs):
        for name in self.column_names:
            df[name + self.suffix] = df[name].rolling(self.window, min_periods=1).min()
        return df


class DataProcessorMean(DataProcessorBase):
    """
    Calculate running mean
    """
    def __init__(self, window_length=None, **kwargs):
        """
        :param window_length: int
        :param kwargs:
        """
        super(DataProcessorMean, self).__init__(**kwargs)
        if type(window_length) is not int:
            raise AttributeError("Window length has to be int")
        self.window = window_length
        if not self.suffix:
            self.suffix = '_mean_' + str(self.window)

    def __call__(self, df, *args, **kwargs):
        for name in self.column_names:
            df[name + self.suffix] = df[name].rolling(self.window, min_periods=1).mean()
        return df
