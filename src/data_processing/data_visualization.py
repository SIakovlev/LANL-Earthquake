import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import dask.dataframe as dd
import dask
import plotly
plotly.tools.set_credentials_file(username='ptolmachev', api_key='Fs5sBFAg7YuBn52rzy6n')

#simple function for plotting small datasets in matplotlib
def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    try:
        plt.plot(series.compute().tolist(), 'r-',linewidth = 2, alpha = 0.7)
    except:
        plt.plot(series.tolist(), 'r-',linewidth = 2, alpha = 0.7)
    plt.show()


