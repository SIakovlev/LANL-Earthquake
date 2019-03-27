import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='ptolmachev', api_key='Fs5sBFAg7YuBn52rzy6n')
import sys
sys.path.insert(0, '../.')
from utils import chunk_data_on_EQs
import os, fnmatch
from dp_utils import *


def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    try:
        plt.plot(series.compute().tolist(), 'r-',linewidth = 2, alpha = 0.7)
    except:
        plt.plot(series.tolist(), 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def plot_data(function, params, **kwargs):
    size_of_slice = 150000
    data_path = kwargs['path_to_data']
    list_of_EQs_to_plot = kwargs['EQs_num']

    if os.path.isdir(data_path) == False:
        raise ValueError("{} is no such directory!".format(data_path))

    # plotting part
    dataframes = [pd.read_hdf(data_path + "/EQ_" + str(EQ) + ".h5", key='table') for EQ in list_of_EQs_to_plot]
    for i in range(len(dataframes)):
        dataframes[i].columns = ["s", "ttf"]
    df = pd.concat(dataframes)
    df.columns = ["s", "ttf"]
    names = [function.__name__, "downsampled_signal", 'ttf']

    # Downsampling is conducted by the last element
    signals = [function(df.s, **params).values.ravel(),  # featurised signal
               w_labels(df.s, **params).values.ravel(),  # downsampled signal
               100 * w_labels(df.ttf, **params).values.ravel()]  # downsample ttf

    s_max = []
    for i in range(len(signals)):
        s_max.append(abs(signals[i]).max())
    s_max.append(0)

    data1 = [go.Scatter(y=(signals[i] - 0.7 * sum(s_max[:i + 1])), opacity=0.7, name=names[i]) for i in
             range(len(signals))]

    layout1 = dict(
        title='Eathquakes ' + function.__name__
    )

    fig1 = dict(data=data1, layout=layout1)
    plotly.offline.plot(fig1, filename="Earthquakes.html", auto_open=True)

    #####################################################################################################
    # takes the first of specified EQs and plots data related to first, middle and last 150000 samples
    b = int((len(dataframes[0].s) - size_of_slice) / 2)
    e = int((len(dataframes[0].s) + size_of_slice) / 2)

    samples = [function(dataframes[0].s[:size_of_slice], **params).values.ravel(),
               function(dataframes[0].s[b:e], **params).values.ravel(),
               function(dataframes[0].s[len(dataframes[0]) - size_of_slice:], **params).values.ravel()]

    s_max = []
    for i in range(len(samples)):
        s_max.append(abs(samples[i]).max())
    s_max.append(0)

    names = ["first " + str(size_of_slice), "middle " + str(size_of_slice), "last " + str(size_of_slice)]
    data2 = [go.Scatter(y=(samples[i] - 0.7 * sum(s_max[:i + 1])), opacity=0.7, name=names[i]) for i in
             range(len(samples))]

    layout2 = dict(
        title='Earthquakes ' + function.__name__
    )

    fig2 = dict(data=data2, layout=layout2)
    plotly.offline.plot(fig2,
                        filename="Earthquakes samples (first, middle, last " + str(size_of_slice) + " dp).html",
                        auto_open=True)

if __name__ == '__main__':
    # example
    params = {"window_size": 1000}
    plot_data(w_std, params, path_to_data="/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/EQs", EQs_num=[2, 3])