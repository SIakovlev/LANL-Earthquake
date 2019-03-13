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


def plot_feature_series(feature_name, params):
    ddfs = [pd.read_hdf("../../data/EQ_" + str(i+1) + ".h5") for i in range(1,4)]
    data = [go.Scatter(y=rolling_window(ddfs[i].s, 1000, 500, feature_name, params = params)+i*500, opacity = 0.7) for i in range(3)]
    layout = dict(
        title='Three Earthquakes ' + feature_name
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename = "Signals before the earthquakes.html", auto_open=True)


#example
#you can define your own fucntion or just pick one from here:
# https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes
#alternatively you may put other fucntions like "np.std"
# example 1
plot_feature_series("quantile", {"q" : 0.6})
# example 2
plot_feature_series("longest_strike_above_mean", None)
# example 3
plot_feature_series("number_crossing_m", {"m" : 0})
