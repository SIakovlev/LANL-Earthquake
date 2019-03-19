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
        self.df = pd.DataFrame()

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
    def __init__(self, **kwargs):
        super(DataProcessorMin, self).__init__(**kwargs)
        if not self.suffix:
            self.suffix = '_min_'

    def __call__(self, df, *args, **kwargs):
        for name in self.column_names:
            temp = df[name]
            # df[name + self.suffix] = df[name].rolling(self.window, min_periods=1).min()
        return df

def calc_win_size():
    pass

def autoscale():
    pass



class DataProcessorMean(DataProcessorBase):
    """
    Calculate running mean
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        """
        super(DataProcessorMean, self).__init__(**kwargs)
        if not self.suffix:
            self.suffix = '_mean_'

    def __call__(self, df, *args, **kwargs):
        for name in self.column_names:
            pass
            # df[name + self.suffix] = df[name].rolling(self.window, min_periods=1).mean()
        return df

