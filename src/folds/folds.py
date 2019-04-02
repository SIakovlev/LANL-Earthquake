import numpy as np

class CustomFold:
    def __init__(self, n_splits=10, shuffle=True, fragmentation=0.1, pad=15):
        if not isinstance(n_splits, int):
            raise ValueError("n_splits should be integer value")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.frag = fragmentation
        self.pad = pad

    def split(self, data):
        for i in range(self.n_splits):
            data_len = data.shape[0]

            # how many samples in test subset
            test_len = int(data_len / self.n_splits)

            # how many consequitive chunks in test subset
            num_test_fragments = max(1, int(test_len * self.frag))
            seq_lens = np.random.rand(num_test_fragments)
            sum_seq_len = test_len / sum(seq_lens)
            seq_lens = [max(1, int(seq_len * sum_seq_len)) for seq_len in seq_lens]

            test_idx = []
            test_idx_padded = []

            for seq_len in seq_lens:
                seq_begin_idx = np.random.randint(0, data_len-1, 1)
                seq_end_idx = min(seq_begin_idx + seq_len, data_len-1)
                seq_end_idx_padded = min(seq_end_idx + self.pad, data_len-1)
                seq_idx_padded = np.arange(seq_begin_idx, seq_end_idx_padded)
                seq_idx = np.arange(seq_begin_idx, seq_end_idx)
                test_idx.extend(seq_idx)
                test_idx_padded.extend(seq_idx_padded)

            train_idx = np.setxor1d(np.arange(0, data_len), test_idx_padded)
            test_idx = np.array(test_idx)

            yield train_idx, test_idx


if __name__ == '__main__':

    train_data = np.arange(int(1000))

    kwargs = {"n_splits": 10, "shuffle": True, "fragmentation": 0.1, "pad": 5}

    folds = CustomFold(**kwargs)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
        # X_train, X_valid = train_data[train_index], train_data[valid_index]

        all_idx = np.arange(0, train_data.shape[0])

        split_union = set(train_index) | set(valid_index)

        diff = set(all_idx).difference(split_union)

        print("Diff between union(train_idx, test_idx) and all_data_idx")
        # print("Diff", diff)
        print("Diff len", len(diff))
        print("Train len", len(train_index))
        print("Test len", len(valid_index))
        # print("Test", valid_index)
        print()

