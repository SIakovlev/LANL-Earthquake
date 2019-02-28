import pandas as pd
import os
import warnings
import glob
import re

VALUE_SIZE = 8


class Stash:
    n_rows = 0
    last_filename = None
    row_size = 0


class DFHandler:

    def __init__(self, **kwargs):

        # TODO: change logic, pass path to a folder where dataframe is

        # self.df_iterator = pd.read_hdf(path, key='table', chunksize=kwargs['chunk_size'])
        # self.file_handler = h5py.File(path)
        self.df_iterator = None
        self.stash = Stash()  # in bytes
        self.file_counter = 0
        self.chunk_size = kwargs['chunk_size']

    def set_iterator(self, path):
        self.df_iterator = list(self.gen(path))

    def set_hdf_iterator(self, path, **kwargs):
        self.df_iterator = pd.read_hdf(path, key='table', chunksize=self.chunk_size)

    def __iter__(self):
        try:
            return iter(self.df_iterator)
        except ValueError:
            print("Iterator is not specified")

    def __getitem__(self, loc):
        """

        :param loc: row location
        :type loc: slice, int
        :return:
        :rtype:
        """
        if isinstance(loc, slice):
            # TODO: handling for a slice object:
            print(loc.start, loc.stop, loc.step)
        else:
            # TODO: handling for a plain index
            print(loc)

    def iloc(self, loc):
        """

        :param loc: column location
        :type loc: slice, list of ints, int,
        :return:
        :rtype:
        """

        if isinstance(loc, slice):
            print()
        elif isinstance(loc, list):
            print()
        elif isinstance(loc, int):
            for df in iter(self.df_iterator):
                df_idx = max(df.index)
                df_length = df.shape[0]
                if loc < df_idx:
                    return df.iloc[df_length + loc - df_idx - 1]
        else:
            raise TypeError("The argument of iloc can only be: slice, list of ints or int")

    def save(self, obj, path, **kwargs):
        """

        :param obj: data frame object
        :type obj: pandas.DataFrame
        :param path: path to a folder where data needs to be saved
        :type path: string
        :param kwargs: chunk_size_MB - data chunk size in MB
        :type kwargs: chunk_size_MB - int
        :return:
        :rtype:
        """

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("You can save pandas DataFrames only")

        if not isinstance(path, str):
            raise TypeError("Path should be specified as a string")

        try:
            os.makedirs(os.path.join(path, kwargs['dir_name']))
        except FileExistsError:
            warnings.warn('Directory {} already exists! This can lead to mixing several datasets into a single one'.
                          format(os.path.join(path, kwargs['dir_name'])))

        save_path = os.path.join(path, kwargs['dir_name'])
        chunk_size_MB = kwargs['chunk_size_MB']
        chunk_size_B = chunk_size_MB * 1e6
        row_size_B = int(obj.shape[1] * VALUE_SIZE)
        chunk_rows = int(chunk_size_B) // row_size_B

        if self.stash.n_rows:
            # if the next file is large enough
            if self.stash.n_rows < obj.shape[0]:
                obj.iloc[:chunk_rows-self.stash.n_rows].to_hdf(self.stash.last_filename, key='table', append=True)
                obj = obj.drop(obj.index[[i for i in range(0, chunk_rows-self.stash.n_rows)]])
            else:
                obj.iloc[:].to_hdf(self.stash.last_filename, key='table', append=True)
                return
            self.stash.n_rows = 0

        N, M = obj.shape
        num_chunks = (M * N * VALUE_SIZE) // int(chunk_size_B)
        # Store the whole object in chunks of size chunk_size
        for i in range(num_chunks):
            filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            obj.iloc[i * chunk_rows: (i + 1) * chunk_rows].to_hdf(filename, key='table')
            self.file_counter += 1

        # Calculate the rest data size in MB and put in in stash
        self.stash.n_rows = N - (chunk_rows * num_chunks)
        if self.stash.n_rows:
            last_filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            obj.iloc[num_chunks * chunk_rows:].to_hdf(last_filename, key='table', append=True)
            self.stash.last_filename = last_filename
            self.file_counter += 1

        # TODO: add data integrity check

    def check_integrity(self, **kwargs):
        size_list = []
        indices = []
        for df in iter(self.df_iterator):
            size_list.append(df.shape[0])
            indices.extend([df.index[0], df.index[-1]])

        # check size
        check_size = size_list[1:-1] == size_list[:-2]
        # check indices
        check_indices = []
        for i in range(0, len(indices) - 3, 2):
            check_indices.append(indices[i+1] + 1 == indices[i+2])
        # check total size
        return check_size, all(check_indices)

    @staticmethod
    def gen(directory):
        file_list = glob.glob(os.path.join(directory, '*.h5'))
        file_list.sort(key=natural_keys)
        for filename in file_list:
            yield pd.read_hdf(filename, key='table')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    filename = re.sub('.h5$', '', text.split('/')[-1])  # remove .h5 extention
    return [atoi(c) for c in re.split(r'(\d+)', filename)]
