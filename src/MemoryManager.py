import pandas as pd
import os
import warnings
import glob
from utils import natural_keys

VALUE_SIZE = 8


class Stash:
    num_rows = 0
    last_filename = None


class MemoryManager:

    def __init__(self, **kwargs):
        """
        Initialise MemoryManager constructor
        :param kwargs:
        :type kwargs:
        """
        self.df_iterator = None         # data frame iterator (set by a separate method set_iterator())
        self.stash = Stash()            # in bytes
        self.file_counter = 0           # counter for saving purposes

    def set_iterator(self, path):
        """
        Create an iterator over objects in a given directory containing .h5 files. Do some preparation
        :param path: path to the directory with .h5 files
        :type path: str
        :return: None
        :rtype: None
        """
        self.df_iterator = list(self.gen(path))                 # set iterator
        self.file_counter = 0                                   # reset file counter
        self.chunk_nrows = self.df_iterator[0].shape[0]         # calculate number of rows in a single chunk

    # TODO: fix later, will be deprecated for sure
    def set_hdf_iterator(self, path, **kwargs):
        self.df_iterator = pd.read_hdf(path, key='table', chunksize=kwargs['chunk_size'])

    def __iter__(self):
        try:
            return iter(self.df_iterator)
        except ValueError:
            print("Iterator is not specified")

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

        # 1) Create a new directory for a shredded dataset
        # --------------------------------------------------------------------------------------
        save_path = os.path.join(path, kwargs['dir_name'])
        if not os.path.exists(save_path):
            if not self.file_counter:
                os.makedirs(save_path)

        chunk_size_B = kwargs['chunk_size_MB'] * 1e6        # convert chunk size to bytes
        row_size_B = int(obj.shape[1] * VALUE_SIZE)         # calculate row size in bytes
        chunk_rows = int(chunk_size_B) // row_size_B        # calculate number of rows per chunk

        # 2) Check if stash is not empty
        # --------------------------------------------------------------------------------------
        if self.stash.num_rows:
            # check if the next file is larger then the new dataframe (this might happen in case it's the last
            # data frame in the directory, which might have an arbitrary small size.
            if self.stash.num_rows < obj.shape[0]:
                obj.iloc[:chunk_rows-self.stash.num_rows].to_hdf(self.stash.last_filename, key='table', append=True)
                # remove elements that are already stored
                obj = obj.drop(obj.index[[i for i in range(0, min(chunk_rows - self.stash.num_rows, obj.shape[0]))]])
            else:
                obj.iloc[:].to_hdf(self.stash.last_filename, key='table', append=True)
                return
            # clean stash
            self.stash.num_rows = 0

        # 3) Calculate the number of chunks and store them
        # --------------------------------------------------------------------------------------
        N, M = obj.shape
        num_chunks = (M * N * VALUE_SIZE) // int(chunk_size_B)

        # check if the chunk size is larger than a dataframe that is stored
        # if so, notify user about it with a warning
        if not num_chunks:
            warnings.warn('Chunk size is too large, please choose a smaller value')
            print('Dataframe size is {} MB'.format(M * N * VALUE_SIZE / int(1e6)))
            print('Chunk size is {} MB'.format(chunk_size_B / int(1e6)))

        for i in range(num_chunks):
            filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            if os.path.exists(filename):
                warnings.warn('Rewriting the existing file: {}'.format(filename))
            obj.iloc[i * chunk_rows: (i + 1) * chunk_rows].to_hdf(filename, key='table')
            self.file_counter += 1

        # 4) Calculate the rest data size in MB and then:
        # - put it in stash
        # - store in the file which name should be remembered as well for the next function call
        # --------------------------------------------------------------------------------------
        self.stash.num_rows = N - (chunk_rows * num_chunks)
        if self.stash.num_rows:
            last_filename = os.path.join(save_path, 'part_{}.h5'.format(self.file_counter))
            if os.path.exists(last_filename):
                warnings.warn('Rewriting the existing file: {}'.format(last_filename))
            obj.iloc[num_chunks * chunk_rows:].to_hdf(last_filename, key='table', append=True)
            self.stash.last_filename = last_filename
            self.file_counter += 1

    def check_integrity(self, path, **kwargs):
        size_list = []          # list of chunk sizes (in rows)
        indices = []            # list of the first and last indices for each chunk of data
        self.set_iterator(os.path.join(path, kwargs['dir_name']))
        for df in iter(self.df_iterator):
            size_list.append(df.shape[0])
            indices.extend([df.index[0], df.index[-1]])

        # Test 1. All parts except the last one should have the same size
        check_size = size_list[1:-1] == size_list[:-2]
        print()
        print("Data integrity check:")
        # check size
        print("Test 1. All parts except the last one should have the same size: {}".
              format(check_size))

        # Test 2. Indices of two consecutive chunks should differ by 1
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
