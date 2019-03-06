import numpy as np
import pandas as pd
import h5py
import dask.dataframe as dd
import dask
import tsfresh

def rolling_window(series, window_size, stride, function, params = None):
    len_series = len(series)
    num_iter = int(np.ceil((len_series-window_size)/stride)+1) 
    
    if params is None:
        kwargs = {}
    else:
        kwargs = params
    
    shape = series.shape[:-1] + (series.shape[-1] - window_size + 1, window_size)
    strides = series.strides + (series.strides[-1],)
    iterator = iter(np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)[::stride])
    res = 0*np.empty(num_iter, dtype = np.float)
    
    if hasattr(tsfresh.feature_extraction.feature_calculators, function):
        modifier = "tsfresh.feature_extraction.feature_calculators."
    else:
        modifier = ""

    expression = modifier + function + "(next(iterator), **kwargs)"
    
    for i in range(num_iter):
        try:
            res[i] = np.float(eval(expression))
        except StopIteration:
            return res[:i]
    
    return res


def calc_data(df, list_of_functions, list_of_params, window_sizes, stride, save_to):
    '''
    Input: pandas array with columns 's' and 'y' 
    ('s' corresponds to time series and 'y' - to the regression label)
    
    saves new pandas dataframe in hdf5 extension to "save_to" location
    where the columns are the calculated over the time series features from the "list_of_function"
    using windows from "window_sizes" and having a spesified stride
    '''
    
    #checks
    if (len(list_of_functions) != len(list_of_params)) or (len(list_of_params) != len(window_sizes)):
        raise ValueError("Parameters \"list_of_functions\", \
        \"list_of_params\" and \"window_sizes\" must have the same lengths!")
    if stride <=0 :
        raise ValueError("The \"stride\" has to be a postivie number!")
    
    try:
        feature_df = pd.read_hdf(save_to) # if there already exists file with features
    except:
        feature_df = pd.DataFrame()
    
    num_features = len(list_of_functions)
    series = np.array(df["s"], dtype = np.float)
    for i in range(num_features):
        function = list_of_functions[i].split("*")[0]
        window = window_sizes[i]
        print("Calculating function \"{}\" with params: \"{}\" over the window: \"{}\""\
              .format(function, list_of_params[i], window))
        
        name_of_col = list_of_functions[i] + "_" + str(window)
        if name_of_col in list(feature_df.columns):
            pass
        else:
            res = rolling_window(series, window_sizes[i], stride, function, params = list_of_params[i])
            feature_df[name_of_col] = res
            
    feature_df.to_hdf(save_to, key='df')        
    return None


#EXAMPLE
df = (pd.read_hdf('~./data/sample.h5', key = 'df'))
df.columns = ["s","y"]
#if one needs to calculate the same feature twice (althoug with different args) one may use the following syntaxis: function*unique_modifier_1, function**unique_modifier_2
list_of_functions = ["np.max",'np.min', "abs_energy","np.std", \
                     "quantile*1", "quantile*2", "mean_second_derivative_central"] 
list_of_params = [None, None, None, None, {"q" : 0.6}, {"q" : 0.8}, None]
window_sizes = len(list_of_functions)*[1000]

stride = 500
save_to = "~./data/featurized_sample.h5"
calc_data(df, list_of_functions, list_of_params, window_sizes, stride, save_to)
featurised_df = pd.read_hdf(save_to, key = 'df')
featurised_df.head()
