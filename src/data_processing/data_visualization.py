import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='ptolmachev', api_key='Fs5sBFAg7YuBn52rzy6n')
import sys
from dp_features import *


def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    try:
        plt.plot(series.compute().tolist(), 'r-',linewidth = 2, alpha = 0.7)
    except:
        plt.plot(series.tolist(), 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def plot_data(list_of_dataframes, feature_name, window_size):

    if not np.all([dataframe.shape[1] == list_of_dataframes[0].shape[1] for dataframe in list_of_dataframes] ):
        raise ValueError("The number of columns in dataframes in 'list_of_dataframes' should be the same! ")

    size_of_slice = int(150000/window_size)
    num_EQs = len(list_of_dataframes)

    # plotting part
    for i in range(num_EQs):
        list_of_dataframes[i].columns = [feature_name,"s", "ttf"]

    df = pd.concat(list_of_dataframes)
    names = [feature_name, "downsampled_signal", 'downsampled_ttf']
    df.columns = names

    # Downsampling is conducted by the last element
    signals = [df[feature_name].ravel(),  # featurised signal
               df["downsampled_signal"].ravel(),  # downsampled signal
               df["downsampled_ttf"].ravel()]  # downsample ttf

    s_max = []
    for i in range(len(signals)):
        s_max.append(abs(signals[i]).max())
    s_max.append(0)

    data1 = [go.Scatter(y=(signals[i] - 0.7 * sum(s_max[:i + 1])), opacity=0.7, name=names[i]) for i in
             range(len(signals))]

    layout1 = dict(
        title='Eathquakes ' + feature_name
    )

    fig1 = dict(data=data1, layout=layout1)
    plotly.offline.plot(fig1, filename="Earthquakes.html", auto_open=True)

    #####################################################################################################
    # takes the first of specified EQs and plots data related to first, middle and last 150000 samples
    b = int((len(list_of_dataframes[0].s) - size_of_slice) / 2)
    e = int((len(list_of_dataframes[0].s) + size_of_slice) / 2)

    samples = [list_of_dataframes[0][feature_name][:int(size_of_slice)].values.ravel(),
               list_of_dataframes[0][feature_name][b:e].values.ravel(),
               list_of_dataframes[0][feature_name][len(list_of_dataframes[0]) - size_of_slice:].values.ravel()]

    s_max = []
    for i in range(len(samples)):
        s_max.append(abs(samples[i]).max())
    s_max.append(0)

    names = ["first " + str(size_of_slice), "middle " + str(size_of_slice), "last " + str(size_of_slice)]
    data2 = [go.Scatter(y=(samples[i] - 0.7 * sum(s_max[:i + 1])), opacity=0.7, name=names[i]) for i in
             range(len(samples))]

    layout2 = dict(
        title='Earthquakes ' + feature_name
    )

    fig2 = dict(data=data2, layout=layout2)
    plotly.offline.plot(fig2,
                        filename="Earthquakes samples (first, middle, last " + str(size_of_slice) + " dp).html",
                        auto_open=True)

if __name__ == '__main__':
    # example
    #list of dataframes
    list_of_dataframes = []
    window_size = 15000
    feature_name = 'quantile'
    params = {'q' : 0.05}
    for i in [3,4,5,7]: # which eqs to take
        df = pd.read_hdf('../../data/EQs/EQ_'+str(i)+'.h5', key='table')
        feature = eval("w_"+feature_name + "(df.s, window_size = window_size, **params)")
        downsampled_s = w_last_elem(df.s, window_size = window_size)
        downsampled_ttf = w_last_elem(df.ttf, window_size = window_size)
        processed_df = pd.DataFrame(np.array([feature.values.ravel(), downsampled_s.values.ravel(), downsampled_ttf.values.ravel()]).T)
        processed_df.columns = [feature_name,"downsampled_signal", "downsampled_ttf"]
        list_of_dataframes.append(processed_df)
    plot_data(list_of_dataframes, feature_name, window_size)