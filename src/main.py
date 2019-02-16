import json
import matplotlib as mpl
import platform
import importlib
import numpy as np
import pandas as pd


if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific

# Load packages
from DataProcessing import DataProcessorBase, DataProcessorMin

# from FeatureEngineering import FE_unit

# from Validation import

def main(**kwargs):

    # Create data processor object, convert to hdr5, load it if exists
    data_processor = DataProcessorBase()
    data_processor_min = DataProcessorMin(**{'cell_names': ['acoustic_data'], 'window_length': 100})
    # train_data = data_processor.data_loader('~/Dev/Kaggle/LANL-Earthquake-Prediction/train.csv')
    df = data_processor.load('~/Dev/Kaggle/LANL-Earthquake-Prediction/train.h5')
    df = data_processor_min.process(df)
    data_processor.save(df, '~/Dev/Kaggle/LANL-Earthquake-Prediction/train_processed.h5')

    # Explore data
    # eda_test = EDA_unit()


    # Feature engineering
    # fe_unit = FE_unit()


if __name__ == '__main__':

    # with open('../settings.json') as settings:
    #     params = json.load(settings)

    main()
