import json
import argparse
import pandas as pd
import platform
import matplotlib as mpl
import copy
import pickle
import os
import re
import sys
import glob
import pywt
import gc
from tqdm import tqdm
from src.utils import str_to_class
from src.validation.validation_utils import read_write_summary
import subprocess
from sklearn.externals import joblib

if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific

# def wavelet_s(df, window_size, wavelet_name):
#     temp = []
#     for i in tqdm(range(0, df.shape[0], window_size)):
#         x = df[i: i + window_size]
#         if len(x) < window_size:
#             [(cA2, _)] = pywt.swt(x, wavelet_name, level=1)
#         else:
#             (cA2, _), (_, _) = pywt.swt(x, wavelet_name, level=2)
#         temp.append(cA2)
#         del x, cA2
#         gc.collect()
#
#     return temp


def main(**kwargs):
    sort_test_names = os.listdir(kwargs['test_dir'])
    sort_test_names = [x for x in sort_test_names if x.endswith('.csv')]
    sort_test_names.sort(key=lambda x: (os.path.isdir(x), x))


    #load model
    model_path = os.path.join(kwargs["models_dir"], kwargs["model_fname"])
    if kwargs["model_load"] == 'sklearn':
        _, file_extension = os.path.splitext(model_path)
        model_class = read_write_summary(model_path, file_extension, 'rb')
    else:
        with open(model_path, 'rb') as f:
            model_class = str_to_class(__name__, kwargs['model_load'])
            model_class = model_class.load(f)

    preprocessor_class = None
    #load preprocess
    if "preproc_path" in kwargs.keys():
        preprocessor_class = read_write_summary(os.path.join(kwargs["models_dir"], kwargs["preproc_path"]), '.pickle', 'rb')


    df_to_submit = pd.DataFrame(columns=["seg_id", "time_to_failure"])

    # forbid subprocess print in stdout
    devnull = open(os.devnull, 'w')

    #create path to test config
    config_dp = os.path.basename(kwargs["dp_config"])
    config_test_dp = re.sub(config_dp, 'test_dp_config.json', kwargs["dp_config"])

    #retrieve dictionary from dp_config
    with open(kwargs["dp_config"], "r") as read_file:
        test_config = json.load(read_file)
        test_config["data_dir"] = kwargs['test_dir']
        test_config["data_processed_dir"] = kwargs["output_dir"]


    for _, test_file in enumerate(tqdm(sort_test_names)):

        test_path = os.path.join(kwargs["test_dir"], test_file)

        #modify test_config
        test_config["data_fname"] = test_file
        test_config["data_processed_fname"] = f"{test_file.split('.')[0]}.h5"

        with open(config_test_dp, 'w') as write_file:
            json.dump(test_config, write_file, sort_keys=True, indent=5)

        #preprocess test data with dp
        subprocess.check_call(["python","/home/alex/kaggleLan/LANL-Earthquake/LANL-Earthquake/src/data_processing/dp_run.py", f"--config_fname={config_test_dp}"],
                              stdout=devnull,stderr=devnull)

        #try:
        test_df = pd.read_hdf(os.path.join(kwargs["output_dir"], f"{test_file.split('.')[0]}.h5"),key='table')

        if  preprocessor_class is not None:
            X_test = preprocessor_class.transform(test_df)

        data_predict = model_class.predict(X_test)
        df_to_submit.loc[len(df_to_submit)] = [test_file.split('.')[0], data_predict[-1]]
       # except UnboundLocalError:
        #    raise ValueError("model not trained")

    print("***********************Save to file***************************")
    df_to_submit.to_csv(kwargs['output_file'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str)

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)