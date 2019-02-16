import pandas as pd
import numpy as np

"""
DataProcessing module.

Date: 16.02.2019

Description:







"""

class DataProcessorBase:

    def __init__(self, cell_names=None, **kwargs):
        self.cell_names = cell_names  # ?

    def load(self, path, key='table', **kwargs):

        # TODO: fix for large frames: (generators?)

        return pd.read_hdf(path, key)

    def save(self, obj, path, key='table', **kwargs):

        # TODO: fix for large frames: (generators?)

        obj.to_hdf(path, key)

    def data_loader(self, path, **kargs):
        return pd.read_csv(path, dtype={'s': np.int16, 'ttf': np.float32})

    def __call__(self, df, *args, **kwargs):

        raise NotImplementedError

        return df


class DataProcessorMin(DataProcessorBase):
    def __init__(self, **kwargs):
        super(DataProcessorMin, self).__init__(**kwargs)
        self.window = kwargs['window_length']  # ?

    def __call__(self, df, *args, **kwargs):

        # TODO: fix for large frames: (generators?)

        for name in self.cell_names:
            df[name + '_min_' + str(self.window)] = df[name].rolling(self.window, min_periods=1).min()
        return df


class DataProcessorMean(DataProcessorBase):
    def __init__(self, **kwargs):
        super(DataProcessorMean, self).__init__(**kwargs)
        self.window = kwargs['window_length']

    def __call__(self, df, *args, **kwargs):
        for name in self.cell_names:
            df[name + '_mean_' + str(self.window)] = df[name].rolling(self.window, min_periods=1).mean()
        return df
