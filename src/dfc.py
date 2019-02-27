import pandas as pd
import h5py
import os
import warnings
import glob

VALUE_SIZE = 8


class Stash:
    B = 0
    last_filename = None


class DFHandler:

    def __init__(self):

        # TODO: change logic, pass path to a folder where dataframe is

        # self.df_iterator = pd.read_hdf(path, key='table', chunksize=kwargs['chunk_size'])
        # self.file_handler = h5py.File(path)
        self.df_iterator = None
        self.stash = Stash()  # in bytes
        self.file_counter = 0

    def set_iterator(self, path):
        self.df_iterator = self.gen(path)

    def set_hdf_iterator(self, path, **kwargs):
        self.df_iterator = pd.read_hdf(path, key='table', chunksize=kwargs['chunk_size'])

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
            print()
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

        if self.stash.B:
            N, M = obj.shape
            stash_size_rows = int(self.stash.B) // int(M * VALUE_SIZE)
            chunk_size_rows = int(chunk_size_B) // int(M * VALUE_SIZE)
            # if the next file is large enough
            if self.stash.B < N:
                obj.iloc[:chunk_size_rows-stash_size_rows].to_hdf(self.stash.last_filename, key='table', append=True)
                obj = obj.drop(obj.index[[i for i in range(0, chunk_size_rows-stash_size_rows)]])
            else:
                obj.iloc[:N].to_hdf(self.stash.last_filename, key='table', append=True)
                return
            self.stash.B = 0

        N, M = obj.shape
        num_chunks = (M * N * VALUE_SIZE) // int(chunk_size_B)
        chunk_size_rows = int(chunk_size_B) // int(M * VALUE_SIZE)

        for i in range(num_chunks):
            filename = os.path.join(save_path, 'part_{}_{}.h5'.format(self.file_counter, i))
            obj.iloc[i * chunk_size_rows: (i + 1) * chunk_size_rows].to_hdf(filename, key='table')

        # Calculate the rest data size in MB and put in in stash
        self.stash.B = N - (chunk_size_rows * num_chunks)
        if self.stash.B:
            last_filename = os.path.join(save_path, 'part_{}_{}.h5'.format(self.file_counter, num_chunks))
            obj.iloc[num_chunks * chunk_size_rows:].to_hdf(last_filename, key='table', append=True)
            self.stash.last_filename = last_filename

        # calculate its size in bytes
        self.file_counter += 1

        # TODO: add data integrity check

    @staticmethod
    def gen(directory):
        file_list = sorted(glob.glob(os.path.join(directory, '*.h5')))
        for filename in file_list:
            yield pd.read_hdf(filename, key='table')
