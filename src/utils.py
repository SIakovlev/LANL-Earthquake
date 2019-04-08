import importlib
import pandas as pd
import os
import numpy as np


def str_to_class(module_name, class_name):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            raise AttributeError(f'Class does not exist: {class_name}')
    except ImportError:
        raise ImportError(f'Module does not exist: {module_name}')
    return class_ or None

def csv_to_h5(fname):
    df = pd.read_csv(fname)
    df.rename(columns={"acoustic_data": "s", "time_to_failure": "ttf"}, inplace=True)
    new_fname = os.path.splitext(fname)[0] + ".h5"
    df.to_hdf(new_fname, key='table')

# returns the indices of zeros of ttf
def get_indices_of_zeros(x):
    indices = np.where(x < 0.1)[0]
    if len(indices) == 0:
        raise ValueError("The data provided contains no full Earthquake!")
    zeros = [0]
    for i in range(len(indices)):
        if (indices[i] - indices[i-1]) > 100000:
            zeros.append(indices[i-1])
    zeros.append(indices[i])
    return zeros

def chunk_data_on_EQs(path_to_data, save_to):
    '''
    :param path_to_data: str, directory with either train.h5 or train.csv
    :param save_to: str, specifies where to save the data
    :return: None
    '''
    if os.path.isfile(path_to_data) == False:
        raise  IOError("Incorrectly specified path or no such file!")

    if os.path.isdir(save_to) == False:
        try:
            os.mkdir(save_to)
        except OSError:
            print("Creation of the directory %s failed" % save_to)
        else:
            print("Successfully created the directory %s " % save_to)

    try:
        df = pd.read_hdf(path_to_data, key = 'table')
    except:
        df = pd.read_csv(path_to_data)

    print(df.info(memory_usage='deep'))
    print("The data is loaded into the memory.")
    df.columns = ['s', 'ttf']
    x = np.array(df.ttf)
    EQs_time = get_indices_of_zeros(x)

    for i in range(len(EQs_time)-1):
        print("\rSaving the data relevant to EQ_{}".format(i+1))
        df.iloc[EQs_time[i]:EQs_time[i+1]].to_hdf(save_to + "/EQ_"+str(i+1)+".h5", key = "table", mode = 'w')

if __name__ == '__main__':
    path_to_data = "../data/train.csv"
    save_to = "../data/EQs"
    chunk_data_on_EQs(path_to_data,save_to)