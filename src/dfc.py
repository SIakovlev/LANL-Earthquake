import pandas as pd
import os
import warnings
import glob
import re

VALUE_SIZE = 8


class Stash:
    num_rows = 0
    last_filename = None


class MemoryManager:

    def __init__(self, **kwargs):
        """

        :param kwargs: chunk_size -
        :type kwargs:
        """
        # TODO: finish description

        # self.df_iterator = pd.read_hdf(path, key='table', chunksize=kwargs['chunk_size'])
        # self.file_handler = h5py.File(path)
        self.df_iterator = None
        self.stash = Stash()  # in bytes
        self.file_counter = 0
        self.chunk_size = kwargs['chunk_size']

        # self.num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

    def set_iterator(self, path):
        self.df_iterator = list(self.gen(path))
        self.nfiles = len(self.df_iterator)
        self.chunk_nrows = self.df_iterator[0].shape[0]
        # last chunk can have a different number of rows due to arbitrary chunk size
        self.last_chunk_nrows = self.df_iterator[-1].shape[0]

    # not sure how to embed yet
    # TODO: fix later, will be deprecated for sure
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
        if self.df_iterator is None:
            raise Exception('Memory manager iterator is not set!')

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
            if loc.step is not None:
                raise Exception("Slicing with a fixed step size is not supported yet")

            start_chunk_number = loc.start // self.chunk_nrows
            start_index = loc.start % self.chunk_nrows
            stop_chunk_number = loc.stop // self.chunk_nrows
            stop_index = loc.stop % self.chunk_nrows

            if start_chunk_number == stop_chunk_number:
                return self.df_iterator[start_chunk_number].iloc[start_index:stop_index]
            elif start_chunk_number > stop_chunk_number:
                df = pd.DataFrame()
                df.append(self.df_iterator[start_chunk_number].iloc[start_index:])
                for chunk_number in range(start_chunk_number + 1, stop_chunk_number):
                    df.append(self.df_iterator[chunk_number])
                df.append(self.df_iterator[stop_chunk_number].iloc[:stop_index])
                return df

        elif isinstance(loc, list):
            df = pd.DataFrame()
            for loc_i in loc:
                chunk_number = loc_i // self.chunk_nrows
                index = loc_i % self.chunk_nrows
                df = df.append(self.df_iterator[chunk_number].iloc[index])
            return df

        elif isinstance(loc, int):
            chunk_number = loc // self.chunk_nrows
            index = loc % self.chunk_nrows
            return self.df_iterator[chunk_number].iloc[index]

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
        chunk_size_B = kwargs['chunk_size_MB'] * 1e6
        row_size_B = int(obj.shape[1] * VALUE_SIZE)
        chunk_rows = int(chunk_size_B) // row_size_B

        if self.stash.num_rows:
            # if the next file is large enough
            if self.stash.num_rows < obj.shape[0]:
                obj.iloc[:chunk_rows-self.stash.num_rows].to_hdf(self.stash.last_filename, key='table', append=True)
                obj = obj.drop(obj.index[[i for i in range(0, chunk_rows - self.stash.num_rows)]])
            else:
                obj.iloc[:].to_hdf(self.stash.last_filename, key='table', append=True)
                return
            self.stash.num_rows = 0

        N, M = obj.shape
        num_chunks = (M * N * VALUE_SIZE) // int(chunk_size_B)
        # Store the whole object in chunks of size chunk_size
        for i in range(num_chunks):
            filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            obj.iloc[i * chunk_rows: (i + 1) * chunk_rows].to_hdf(filename, key='table')
            self.file_counter += 1

        # Calculate the rest data size in MB and put in in stash
        self.stash.num_rows = N - (chunk_rows * num_chunks)
        if self.stash.num_rows:
            last_filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            obj.iloc[num_chunks * chunk_rows:].to_hdf(last_filename, key='table', append=True)
            self.stash.last_filename = last_filename
            self.file_counter += 1

    def check_integrity(self, **kwargs):
        size_list = []
        indices = []
        for df in iter(self.df_iterator):
            size_list.append(df.shape[0])
            indices.extend([df.index[0], df.index[-1]])

        check_size = size_list[1:-1] == size_list[:-2]
        print()
        print("Data integrity check:")
        # check size
        print("Test 1. All parts except the last one should have the same size: {}".
              format(check_size))

        # check indices
        check_indices = []
        for i in range(0, len(indices) - 3, 2):
            check_indices.append(indices[i+1] + 1 == indices[i+2])

        print("Test 2. Indices of two consecutive chunks should differ by 1: {}".
              format(all(check_indices)))
        print()

        return check_size and all(check_indices)

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
