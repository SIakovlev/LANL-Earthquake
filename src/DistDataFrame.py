import pandas as pd
import glob
from utils import natural_keys
import os


class iLocWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, loc):
        """

        :param loc: position index for a distributed dataframe
        :type loc: slice, list of ints or int
        :return: part of the dataframe specified by loc
        :rtype: pd.DataFrame
        """
        if isinstance(loc, slice):
            if loc.step is not None:
                raise Exception("Slicing with a fixed step size is not supported yet")

            start_chunk_number = loc.start // self.obj.chunk_nrows
            start_index = loc.start % self.obj.chunk_nrows
            stop_chunk_number = loc.stop // self.obj.chunk_nrows
            stop_index = loc.stop % self.obj.chunk_nrows

            if start_chunk_number == stop_chunk_number:
                return self.obj.df_iterator[start_chunk_number].iloc[start_index:stop_index]
            elif stop_chunk_number > start_chunk_number:
                df = pd.DataFrame()
                df = df.append(self.obj.df_iterator[start_chunk_number].iloc[start_index:])
                for chunk_number in range(start_chunk_number + 1, stop_chunk_number):
                    df = df.append(self.obj.df_iterator[chunk_number])
                df = df.append(self.obj.df_iterator[stop_chunk_number].iloc[:stop_index])
                return df

        elif isinstance(loc, list):
            df = pd.DataFrame()
            for loc_i in loc:
                chunk_number = loc_i // self.obj.chunk_nrows
                index = loc_i % self.obj.chunk_nrows
                df = df.append(self.obj.df_iterator[chunk_number].iloc[index])
            return df

        elif isinstance(loc, int):
            chunk_number = loc // self.obj.chunk_nrows
            index = loc % self.obj.chunk_nrows
            return self.obj.df_iterator[chunk_number].iloc[index]

        else:
            raise TypeError("The argument of iloc can only be: slice, list of ints or int")

    def __setitem__(self, loc, value):
        """

        :param loc:
        :type loc:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        if isinstance(loc, slice):
            pass
            # Not sure if implemented properly in pandas

        elif isinstance(loc, list):
            pass
            # Not sure if implemented properly in pandas

        elif isinstance(loc, int):
            chunk_number = loc // self.obj.chunk_nrows
            index = loc % self.obj.chunk_nrows
            self.obj.df_iterator[chunk_number].iloc[index] = value

        else:
            raise TypeError("The argument of iloc can only be: slice, list of ints or int")


class DistDataFrame:

    def __init__(self, path, **kwargs):
        """
        Initialise DistDataFrame constructor
        :param kwargs:
        :type kwargs:
        """
        self.df_iterator = list(self.gen(path))             # data frame iterator (set by a separate method set_iterator()
        self.iloc = iLocWrapper(self)                       # wrapper for .iloc[] (same as for pandas DataFrame)
        self.chunk_nrows = self.df_iterator[0].shape[0]     # calculate number of rows in a single chunk

    def __getitem__(self, loc):
        """
        Allows [] usage for the class object
        :param loc: column name
        :type loc: str
        :return: column specified by loc
        :rtype: pd.DataFrame
        """
        if self.df_iterator is None:
            raise Exception('Memory manager iterator is not set!')

        if isinstance(loc, slice):
            pass
        elif isinstance(loc, str):
            df_temp = pd.DataFrame()
            for df in self.df_iterator:
                df_temp = df_temp.append(df[[loc]])
            return df_temp
        else:
            raise TypeError("The argument of iloc can only be: slice, list of ints or int")

    def __setitem__(self, loc, value):
        """

        :param loc:
        :type loc:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        if self.df_iterator is None:
            raise Exception('Memory manager iterator is not set!')

        if isinstance(loc, slice):
            pass
        elif isinstance(loc, str):
            for df in self.df_iterator:
                df[loc] = value.iloc[df.index]
            return df
        else:
            raise TypeError("The argument of iloc can only be: slice, list of ints or int")

    def __iter__(self):
        try:
            return iter(self.df_iterator)
        except ValueError:
            print("Iterator is not specified")

    def head(self, num):
        return self.df_iterator[0].head(num)

    def tail(self, num):
        return self.df_iterator[-1].tail(num)

    @staticmethod
    def gen(directory):
        file_list = glob.glob(os.path.join(directory, '*.h5'))
        file_list.sort(key=natural_keys)
        for filename in file_list:
            yield pd.read_hdf(filename, key='table')
