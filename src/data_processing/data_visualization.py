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


# TODO: def plot_func(data, func, params)

def plot_feature_series(path_to_data, feature_name, params, window_size, stride, list_of_EQs, downsample_by = 'central_element'):
    dataframes = [pd.read_hdf(path_to_data + "/EQ_" + str(list_of_EQs[i]) + ".h5") for i in range(len(list_of_EQs))]

    df = pd.concat(dataframes)

    data1 = [go.Scatter(y=rolling_window(df.s, window_size, stride, feature_name, params = params), opacity = 0.7, name  = feature_name),
            go.Scatter(y=rolling_window(df.s-600, window_size, stride, downsample_by, params = None), opacity = 0.7, name = "downsampled signal"),
            go.Scatter(y=rolling_window(10*df.y-300, window_size, stride, downsample_by, params = None), opacity = 0.7, name = "ttf*10")]
    layout1 = dict(
        title='Three Earthquakes ' + feature_name
    )

    fig1 = dict(data=data1, layout=layout1)
    plotly.offline.plot(fig1, filename = "Signals before the earthquakes.html", auto_open=True)

    # b = int( (len(dataframes[0].s) - 150000) / 2)
    # e = int( (len(dataframes[0].s) + 150000) / 2)
    # data2 = [go.Scatter(y=rolling_window(dataframes[0].s[:150000], window_size, stride, feature_name, params = params), opacity = 0.7, name  = "first 150000"),
    #         go.Scatter(y=rolling_window(dataframes[0].s[b:e], window_size, stride, feature_name, params = params), opacity = 0.7, name = "middle 150000"),
    #         go.Scatter(y=rolling_window(dataframes[0].s[-150000:] + 1000, window_size, stride, feature_name, params=params), opacity=0.7, name="last 150000")
    #         ]
    # layout2 = dict(
    #     title='Three Earthquakes ' + feature_name
    # )
    #
    # fig2 = dict(data=data2, layout=layout2)
    # plotly.offline.plot(fig2, filename = "Signals before the earthquakes.html", auto_open=True)


if __name__ == "__main__":
    #example
    #you can define your own fucntion or just pick one from here:
    # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes
    #alternatively you may put other fucntions like "np.std"
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
    path_to_data = "/home/pavel/Documents/0Research/Projects/LANL-Earthquake/data/EQs"
    feature_name = "np.std"
    params = None
    window_size = 1000
    stride = 500
    list_of_EQs = [2,3]
    plot_feature_series(path_to_data, feature_name, params, window_size, stride, list_of_EQs) # +