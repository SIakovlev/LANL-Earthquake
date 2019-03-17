import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objs as go
import dask.dataframe as dd
import dask
import plotly
plotly.tools.set_credentials_file(username='ptolmachev', api_key='Fs5sBFAg7YuBn52rzy6n')
from DP_feature_calc import rolling_window

#simple function for plotting small datasets in matplotlib
def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    try:
        plt.plot(series.compute().tolist(), 'r-',linewidth = 2, alpha = 0.7)
    except:
        plt.plot(series.tolist(), 'r-',linewidth = 2, alpha = 0.7)
    plt.show()


def plot_feature_series(path_to_data, feature_name, params, window_size, stride, list_of_EQs, downsample_by = 'central_element'):
    dataframes = [pd.read_hdf(path_to_data + "/EQ_" + str(list_of_EQs[i]) + ".h5") for i in range(len(list_of_EQs))]

    df = pd.concat(dataframes)

    signals = [rolling_window(df.s, window_size, stride, feature_name, params = params), #featurised signal
               rolling_window(df.s, window_size, stride, downsample_by, params = None), #downsampled signal
               100*rolling_window(df.y, window_size, stride, downsample_by, params = None)] # downsample ttf
    s_max = []
    for i in range(3):
        s_max.append(max(signals[i]))
    s_max.append(0)

    names = [feature_name, "downsampled_signal", 'ttf']
    data1 = [go.Scatter(y=(signals[i] - 0.7*sum(s_max[:i+1])), opacity = 0.7, name  = names[i]) for i in range(3)]

    layout1 = dict(
        title='Eathquakes ' + feature_name
    )

    fig1 = dict(data=data1, layout=layout1)
    plotly.offline.plot(fig1, filename = "Earthquakes.html", auto_open=True)

    #####################################################################################################
    # takes the first of specified EQs and plots data related to first, middle and last 150000 samples
    b = int( (len(dataframes[0].s) - 150000) / 2)
    e = int( (len(dataframes[0].s) + 150000) / 2)

    samples = [rolling_window(dataframes[0].s[:150000], window_size, stride, feature_name, params = params),
               rolling_window(dataframes[0].s[b:e], window_size, stride, feature_name, params = params),
               rolling_window(dataframes[0].s[-150000:], window_size, stride, feature_name, params = params)]

    s_max = []
    for i in range(3):
        s_max.append(max(samples[i]))
    s_max.append(0)

    names = ["first 150000","middle 150000","last 150000"]
    data2 = [go.Scatter(y= (samples[i] - 0.7*sum(s_max[:i+1])), opacity = 0.7, name  = names[i]) for i in range(3)]

    layout2 = dict(
        title='Earthquakes ' + feature_name
    )

    fig2 = dict(data=data2, layout=layout2)
    plotly.offline.plot(fig2, filename = "Earthquakes samples (first, middle, last 150 000 dp).html", auto_open=True)


if __name__ == "__main__":
    #you can define your own fucntion or just pick one from here:
    # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes
    #alternatively you may put other fucntions like "np.std"
    #example
    path_to_data = "/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/EQs"
    feature_name = "quantile"
    params = {"q" : 0.7}
    window_size = 1000
    stride = 500
    list_of_EQs = [3,6]
    plot_feature_series(path_to_data, feature_name, params, window_size, stride, list_of_EQs)


    #some info
    # example 1
    # plot_feature_series("quantile", {"q" : 0.7}) # + gives the results similar to some sort of filtering. There are peaks clearly visible on the graphs
    # # example 2
    # plot_feature_series("longest_strike_above_mean", None) # - mostly seem like a noise
    # # example 3
    # plot_feature_series("number_crossing_m", {"m" : 0}) # - mostly noise

    # plot_feature_series("number_crossing_m", {"m" : 50}) # + seems like it has some information. The further the more variable the signal is

    # plot_feature_series("number_crossing_m", {"m" : 75}) # + seems like it has some information. The further the more variable the signal is

    # plot_feature_series("abs_energy", None) # +- there are a couple of very distinct peakes, but the rest of it seems like noise

    # plot_feature_series("absolute_sum_of_changes", None) # - Doesnt seem any more meaningful than the original signal

    # plot_feature_series("autocorrelation", {"lag":500}) # - BS

    # plot_feature_series("binned_entropy", {"max_bins":50}) # - noise

    # plot_feature_series("binned_entropy", {"max_bins":10}) # - noise

    # plot_feature_series("c3", {"lag":100}) # +- seems very sparse

    # plot_feature_series("c3", {"lag":10}) # +- seems very sparse

    # plot_feature_series("cid_ce", {"normalize":True}) # - there are some weird dips, but overall seems like noise

    # plot_feature_series("count_above_mean", None) # - noise

    # plot_feature_series("np.std", None) # +
