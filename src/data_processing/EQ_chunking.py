import numpy as np
import pandas as pd
import os
pd.set_option('precision', 15)

def cluster(indices):
    zeros = [0]
    for i in range(len(indices)):
        if (indices[i] - indices[i-1]) > 100000:
            zeros.append(indices[i-1])
    zeros.append(indices[i])
    return zeros

def chunk_data_on_EQs(path_to_data, save_to):
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
    df.columns = ['s', 'y']
    x = np.array(df.y)
    indices = np.where(x < 0.1)[0]
    if len(indices) == 0:
        raise ValueError("The data provided contains no full Earthquake!")

    EQs_time = cluster(indices)
    for i in range(len(EQs_time)-1):
        print("\rSaving the data relevant to EQ_{}".format(i+1))
        df.iloc[EQs_time[i]:EQs_time[i+1]].to_hdf(save_to + "/EQ_"+str(i+1)+".h5", key = "table", mode = 'w')

if __name__ == '__main__':

    path_to_data = "/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/train.csv"
    save_to = "/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/EQs"
    chunk_data_on_EQs(path_to_data,save_to)
