from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import PreprocessingBase

class PreprocShuffle(PreprocessingBase):
    """
    Shuffle input dataframes in consistent way
    """

    def __init__(self, sequence_len=1, **kwargs):
        super(PreprocShuffle, self).__init__()
        self.sequence_len = sequence_len

    def fit(self, *dfs):
        pass

    def transform(self, *dfs):
        """
        Shuffle input dataframes in consistent way
        :param dfs: datarframes with consistent first dimension (e.g. (X_train, y_train))
        :return: shuffled dataframes
        """
        num_samples = dfs[0].shape[0]
        num_sequences = num_samples // self.sequence_len
        sequences_order_shuffled = shuffle(np.arange(num_sequences))
        idx_mask = np.array([np.arange(self.sequence_len) + s * self.sequence_len for s in sequences_order_shuffled]).flatten()

        return (df.reindex(idx_mask) for df in dfs)


if __name__ == '__main__':
    num_samples = 100000
    num_cols = 5

    data = np.ones((num_samples, num_cols)) * np.arange(0, num_samples).reshape(-1, 1)
    X_train_df = pd.DataFrame(data, columns=np.arange(0, num_cols))
    y_train_df = pd.DataFrame(data[:, 0] + 5, columns=np.arange(0, 1))

    p_shuffle = PreprocShuffle(sequence_len=1)

    X_shuffled, y_shuffled = p_shuffle.transform(X_train_df, y_train_df)
    print(X_shuffled[:50])
    print(y_shuffled[:50])

    p_shuffle = PreprocShuffle(sequence_len=10)
    X_shuffled, y_shuffled = p_shuffle.transform(X_train_df, y_train_df)
    print(X_shuffled[:50])
    print(y_shuffled[:50])