import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='ptolmachev', api_key='Fs5sBFAg7YuBn52rzy6n')
import sys
from feature import Feature


def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    try:
        plt.plot(series.compute().tolist(), 'r-',linewidth = 2, alpha = 0.7)
    except:
        plt.plot(series.tolist(), 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def plot_data(list_of_dataframes, feature_name, window_size, window_stride):

    if not np.all([dataframe.shape[1] == list_of_dataframes[0].shape[1] for dataframe in list_of_dataframes] ):
        raise ValueError("The number of columns in dataframes in 'list_of_dataframes' should be the same! ")

    size_of_slice = int(150000/window_stride) # 150000 - number of points in one data slice
    num_EQs = len(list_of_dataframes)

    # PLOTTING PART
    for i in range(num_EQs):
        list_of_dataframes[i].columns = [feature_name,"s", "ttf"]

    df = pd.concat(list_of_dataframes)
    names = [feature_name, "downsampled_signal", 'downsampled_ttf']
    df.columns = names

    # Downsampling is conducted by the last element
    signals = [df[feature_name].ravel(),  # featurised signal
               df["downsampled_signal"].ravel(),  # downsampled signal
               df["downsampled_ttf"].ravel()]  # downsample ttf

    traces = []
    for i in range(len(names)): # 3 traces: feature, signal, ttf
        trace = go.Scatter(y=(signals[i]), opacity=0.7, name=names[i])
        traces.append(trace)

    fig1 = tools.make_subplots(rows=len(traces), cols=1)
    for i in range(len(traces)):
        fig1.append_trace(traces[i], i+1, 1)
    plotly.offline.plot(fig1, filename="Earthquakes.html", auto_open=True)

    #####################################################################################################
    # takes the first of specified EQs and plots data related to first, middle and last 150000 samples
    b = int((len(list_of_dataframes[0].s) - size_of_slice) / 2)
    e = int((len(list_of_dataframes[0].s) + size_of_slice) / 2)

    names = ["first " + str(size_of_slice) + 'points',
             "middle " + str(size_of_slice) + 'points',
             "last " + str(size_of_slice) + 'points']

    samples = [list_of_dataframes[0][feature_name][:int(size_of_slice)].values.ravel(),
               list_of_dataframes[0][feature_name][b:e].values.ravel(),
               list_of_dataframes[0][feature_name][len(list_of_dataframes[0]) - size_of_slice:].values.ravel()]
    traces = []
    for i in range(len(names)):
        trace = go.Scatter(y=(samples[i]), opacity=0.7, name=names[i])
        traces.append(trace)

    fig2 = tools.make_subplots(rows=len(traces), cols=1)
    for i in range(len(traces)):
        fig2.append_trace(traces[i], i+1, 1)

    plotly.offline.plot(fig2,
                        filename="Earthquakes samples (first, middle, last " + str(size_of_slice) + " points).html",
                        auto_open=True)

if __name__ == '__main__':
    # example
    list_of_dataframes = []
    window_size = 15000
    window_stride = 1000
    feature_name = 'w_quantile'
    params = {'q' : 0.05}
    for i in [3,4,7]: # which EQs to take
        df = pd.read_hdf('../../data/EQs/EQ_'+str(i)+'.h5', key='table')

        # bunch of feature objects
        f_s = Feature(df['s'], '../../data/EQs/EQ_'+str(i))
        f_downsampled_s = Feature(df['s'], '../../data/EQs/EQ_'+str(i))
        f_downsampled_ttf = Feature(df['ttf'], '../../data/EQs/EQ_'+str(i))

        #get the data
        feature = eval('f_s.' + feature_name + "(df.s, window_size = window_size, window_stride = window_stride, **params).data")
        downsampled_s = f_downsampled_s.w_last_elem(window_size = window_size, window_stride = window_stride).data
        downsampled_ttf = f_downsampled_ttf.w_last_elem(window_size = window_size, window_stride = window_stride).data

        # shape it into a dataframe
        processed_df = pd.DataFrame(np.array([feature.values.ravel(), downsampled_s.values.ravel(), downsampled_ttf.values.ravel()]).T)
        processed_df.columns = [feature_name,"downsampled_signal", "downsampled_ttf"]
        list_of_dataframes.append(processed_df)

    plot_data(list_of_dataframes, feature_name, window_size, window_stride)