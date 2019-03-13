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

print(os.listdir('../../data/'))
df = pd.read_csv('../../data/train.csv')
print(df.info(memory_usage='deep'))
df.columns = ['s', 'y']
x = np.array(df.y)
indices = np.where(x < 0.1)[0]
EQs_time = cluster(indices)

for i in range(len(EQs_time)-1):
    df.iloc[EQs_time[i]:EQs_time[i+1]].to_hdf("../../data//EQ_"+str(i+1)+".h5", key = "df", mode = 'w')