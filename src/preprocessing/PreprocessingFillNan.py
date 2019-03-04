from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import PreprocessingBase

class PreprocFillNa(PreprocessingBase):
    """
    Fill Nans  in input dataframes
    """

    def __init__(self, fill_na_value=0, **kwargs):
        super(PreprocFillNa, self).__init__()
        self.fill_na_value = fill_na_value

    def fit(self, *dfs):
        pass

    def transform(self, *dfs):
        """
        Fill Nans  in input dataframes
        :param dfs: datarframes with consistent first dimension (e.g. (X_train, y_train))
        :return: shuffled dataframes
        """
        return (df.fillna(self.fill_na_value) for df in dfs)


if __name__ == '__main__':
    num_samples = 100
    num_cols = 5

    data = np.ones((num_samples, num_cols)) * np.arange(0, num_samples).reshape(-1, 1)
    data[3:14] = [np.nan]*num_cols

    X_train_df = pd.DataFrame(data, columns=np.arange(0, num_cols))
    y_train_df = pd.DataFrame(data[:, 0] + 5, columns=np.arange(0, 1))

    p_fill_na = PreprocFillNa(sequence_len=1)

    X_fill_na, y_fill_na = p_fill_na.transform(X_train_df, y_train_df)
    print(X_fill_na[:50])
    print(y_fill_na[:50])

