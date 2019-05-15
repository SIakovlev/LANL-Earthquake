import json
import argparse
import numpy as np
import pandas as pd
import platform
import matplotlib as mpl
import copy
import pickle
import os
import re
import sys
import glob
# import pywt
import gc
from subprocess import Popen, PIPE
from tqdm import tqdm
from src.utils import str_to_class
from src.validation.validation_utils import read_write_summary
import subprocess
from sklearn.externals import joblib

from src.models.mlp_net import MLP
from src.models.mlp_classifier_net import MLP_classifier
from src.models.lstm_net import LstmNet
from src.folds.folds import CustomFold
from src.models.nn_test import CustomNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler

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
    if kwargs["model_load"] == 'standard':
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
        test_config["data_processed_dir"] = kwargs["proceded_dir"]

    multi_num = int(kwargs["multi_proc"])
    ## if test shape not divide on multi_num - fix this
    i = 0
    while True:
        if len(sort_test_names)%multi_num == 0:
            break
        elif multi_num - i < 0:
            print(f"Len of test data: {len(sort_test_names)} must divided on multi_num: {multi_num}")
            multi_num = 1
            break
        else:
            i += 1
            multi_num -= i

    print(f'Use {multi_num} processes')

    test_configs = [test_config for x in range(multi_num)]
    config_tests_dp = [config_test_dp[:-5]+str(i)+'.json' for i in range(multi_num)]

    for _, test_files in enumerate(tqdm(np.reshape(sort_test_names, (-1, multi_num)))):

        for i, config_i in enumerate(config_tests_dp):
            #modify test_config
            test_configs[i]["data_fname"] = test_files[i]
            test_configs[i]["data_processed_fname"] = f"{test_files[i].split('.')[0]}.h5"

            with open(config_i, 'w') as write_file:
                json.dump(test_configs[i], write_file, sort_keys=True, indent=5)
        python_dir = kwargs["dp_run_file"]
        cmds_list = [["python", python_dir, f"--config_fname={config_test_dp}"] for config_test_dp in config_tests_dp]
        procs_list = [Popen(cmd, stdout=devnull, stderr=devnull) for cmd in cmds_list]
        for proc in procs_list:
            proc.wait()

        #preprocess test data with dp

        # subprocess.check_call(["python","/home/alex/kaggleLan/LANL-Earthquake/LANL-Earthquake/src/data_processing/dp_run.py", f"--config_fname={config_test_dp}"],
        #                       stdout=devnull, stderr=devnull)

        #try:
        for test_file in test_files:
            test_df = pd.read_hdf(os.path.join(kwargs["proceded_dir"], f"{test_file.split('.')[0]}.h5"), key='table')

            if preprocessor_class is not None:
                print(test_df.head())
                print(preprocessor_class.get_params())
                X_test = pd.DataFrame(preprocessor_class.transform(test_df))
            data_predict = model_class.predict(X_test)
            df_to_submit.loc[len(df_to_submit)] = [test_file.split('.')[0], data_predict[-1].squeeze()]
       # except UnboundLocalError:
        #    raise ValueError("model not trained")
    #remove test configs
    for file_d in config_tests_dp:
        if os.path.exists(file_d):
            os.remove(file_d)
    print("***********************Save to file***************************")

    df_to_submit.to_csv(kwargs['output_file'], index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_fname',
                        help='name of the config file',
                        type=str,
                        default="../configs/multi_test_config.json")

    args = parser.parse_args()

    with open(args.config_fname) as config:
        params = json.load(config)
    main(**params)