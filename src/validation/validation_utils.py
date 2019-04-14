import pandas as pd
import numpy as np

import pickle
import os

import xgboost as xgb

from src.models.nn_test import CustomNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import copy

from src.utils import str_to_class
import re
import ast

import json


"""
ValidationModule

Date: 19.02.2019

Description:

"""


class ValidationBase:

    def __init__(self, **kwargs):

        '''
        Create instance of validation object that will validate models
        :param kwargs:
        '''

        self.score_data = {}
        self.models_list = []
        self.models_features = [{}]
        self._create_models(**kwargs)

    def _reshape_data(self, train_data, y_train_data):
        # DIMENSION OF DATA MUST BE MORE THAN 1
        if train_data.values.ndim == 1:
            train_data = train_data.values.reshape(-1, 1)
            # y_train_data = y_train_data.values.reshape(-1, 1)
        return (train_data, y_train_data)

    def print_models(self):

        '''
        Print models in validator's instance
        '''

        for l_item in self.models_list:
            print(l_item)

    def _create_model(self, model, model_par):
        model = str_to_class(__name__, model)
        return model(**model_par)

    def _create_models(self, **kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = self._create_model(m, m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["model_name"] = str(m)
            self.models_features[i_m]["model_params"] = m_p
            self.models_features.append({})

    def train_models(self, train_data, y_train_data, *args):

        '''
        Train all models in validator's instance
        :param train_data
        :param y_train_data
        :param args: folds, path_to_save, metrics, other parameters that will in summary
        '''

        #reshape if array is 1D
        train_data, y_train_data = self._reshape_data(train_data, y_train_data)

        folds, path_to_save, metric_classes, *params_to_save = args

        fold_data = [x for x in params_to_save if x is not None and 'folds_name' in x.keys()]
        fold_data = fold_data[0]

        #options that not save in output summary
        models_directory_dict = [x for x in params_to_save if x is not None and 'models_directory' in x.keys()]
        predict_data = None
        params_to_save.remove(models_directory_dict[0])

        if models_directory_dict[0]['predict_directory'] is not None:
            predict_data = pd.DataFrame(columns=['y_valid','y_predict'], index= range(train_data.shape[0]))

        for num_model, model in enumerate(self.models_list):
            print("Train model :", self.models_features[num_model]["model_name"])
            for fold_n, (train_index, valid_index) in enumerate(tqdm(folds.split(train_data))):
                # get data
                X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
                y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]
                #clean model
                model = self._create_model(self.models_features[num_model]["model_name"], self.models_features[num_model]["model_params"])
                # train model
                model.fit(X_train, y_train)
                #predict on train
                y_train_pr = model.predict(X_train)
                # validate
                y_predict = model.predict(X_valid)
                if predict_data is not None:
                    predict_data['y_valid'].iloc[valid_index] =y_valid.values
                    predict_data['y_predict'].iloc[valid_index] = np.squeeze(y_predict)
                # compute all scores
                print(f"*******************************Compute on {fold_n} fold*******************************")
                for metric_name, metric_value in metric_classes.items():
                    if fold_n==0:
                        self.score_data[metric_name] = []
                        self.score_data[metric_name+"_train"] = []
                    score_valid = metric_value(y_valid, y_predict)
                    score_train = metric_value(y_train, y_train_pr)
                    print("train_error on METRIC: {0} is {1:0.4f} ".format(metric_name, score_train))
                    print("validation_error on METRIC: {0} is {1:0.4f} ".format(metric_name, score_valid))
                    self.score_data[metric_name + "_train"].append(score_train)
                    self.score_data[metric_name].append(score_valid)
                print("*****************************************************************************************")
            #save predict data
            if predict_data is not None:
                save_predict_path = os.path.join(models_directory_dict[0]['predict_directory'],"{0}_{1}_{2}.pickle".format(
                                                                                    self.models_features[num_model]["model_name"],
                                                                                    self.models_features[num_model]["model_params"],
                                                                                    fold_data["folds_name"]
                                                                                    ))
                predict_data.to_pickle(save_predict_path)

                predict_data = predict_data.iloc[0:train_data.shape[0]]

            #save model
            if models_directory_dict[0]['models_directory'] is not None:
                model_name = "Model_{0}_{1}_train_on_last_{2}.pickle".format(self.models_features[num_model]["model_name"],
                                                                      self.models_features[num_model]["model_params"],
                                                                      fold_data["folds_name"])

                save_model_path = os.path.join(models_directory_dict[0]['models_directory'], model_name)

                read_write_summary(save_model_path, '.pickle', 'wb', model)

            # save summary
            params_to_save.append(self.models_features[num_model])
            # params_to_save.remove(models_directory_dict[0])
            self._save_summary_of_model(path_to_save, params_to_save)
            params_to_save.remove(self.models_features[num_model])
            self.score_data = {}


    def _save_summary_of_model(self, path, *args):

        '''
        Save summary for all models in validator instance
        :param path:
        :param kwargs: columns in summary
        '''

        #that all columns have existed in output
        columns_in_summary = ["data_fname", "preproc_name","preproc_params",
                        "folds_name", "folds_params", "model_name", "model_params"]


        dfObj = pd.DataFrame()
        for s_k,s_v  in self.score_data.items():
            dfObj[s_k] = [s_v]

        keys_in_model = []
        for l in args[0]:
            if l is not None:
                for d_k in l.keys():
                    keys_in_model.append(d_k)
                    dfObj[d_k] = [l[d_k]]

        for d in columns_in_summary:
            if d not in keys_in_model:
                dfObj[d] = np.NaN

        _, file_extension = os.path.splitext(path)

        if not os.path.exists(path):
            read_write_summary(path, file_extension, 'wb', dfObj)
        else:
            #load summary
            modDfObj = read_write_summary(path, file_extension, 'rb')
            #concatenate load summary and new model's summary
            concat = pd.concat([dfObj, modDfObj])
            read_write_summary(path, file_extension, '+wb', concat)
            del concat,modDfObj


def read_write_summary(path, extension, action, df=None):
    '''
    :param path
    :param extension: csv, pickle, ...
    :param action: as like in function: open
    :param df: dataFrame
    :return: return pandas DataFrame
    '''
    if extension == '.pickle':
        with open(path, action) as f:
            if 'r' in action:
                return pickle.load(f)
            else:
                pickle.dump(df, f)
    elif extension == '.csv':
        with open(path, action) as f:
            if 'r' in action:
                return pd.read_csv(path)
            else:
                df.to_csv(path, index=False)
    elif extension == '.h5' or extension == '.hdf5':
        with open(path, action) as f:
            if 'r' in action:
                return pd.read_hdf(path, key='table')
            else:
                df.to_hdf(path, index=False, key='table')
    else:
        raise KeyError(f"extension {extension} not found")


def summary_to_config(path_summary, path_json, rows = 1):

    '''
    :param path_summary:
    :param path_json:
    :param rows: example: 1 or [1,2,3]
    :return: json file in path_json
    '''
    _, file_extension = os.path.splitext(path_summary)
    df = read_write_summary(path_summary, file_extension, 'rb')

    metrics_columns = [f for f in df.columns if not re.match(r"(?:fold|model|data|preproc)", f)]

    dict_json = {}

    dict_json['summary_dest'] = path_summary

    if type(rows) is not list:
        rows = [rows]

    train_data_cont = []

    folds_data = []

    model_data = []

    preproc_data = []

    for l in rows:
        dict_train_data = df['data_fname'].iloc[l]
        if dict_train_data not in train_data_cont:
            train_data_cont.append(dict_train_data)

        if len(train_data_cont) > 1:
            raise AttributeError("Please use models with single train data")

        preproc_name = df['preproc_name'].iloc[l]
        preproc_param = None
        if preproc_name is not None:
            try:
                preproc_param = ast.literal_eval(df['preproc_params'].iloc[l])
                preproc_dict = {"name": preproc_name, **preproc_param}
            except ValueError as e:
                print("not preprocessing params")
                preproc_dict = {"name": preproc_name}
            finally:
                if preproc_dict not in preproc_data:
                    preproc_data.append(preproc_dict)
                if len(preproc_data) > 1:
                    raise AttributeError("Please use models with single preprocessing")

        folds_params = ast.literal_eval(df['folds_params'].iloc[l])
        if type(folds_params) is dict:
            folds_elem = {"name":df['folds_name'].iloc[l],**folds_params}
            if folds_elem not in folds_data:
                folds_data.append(folds_elem)

        model_params = ast.literal_eval(df['model_params'].iloc[l])

        if type(model_params) is dict:
            model_elem = {df['model_name'].iloc[l]: model_params}
            model_dict= {'model':model_elem}
            if model_dict not in model_data:
                model_data.append(model_dict)

    if preproc_data!=[]:
        dict_json['preproc'] = preproc_data[0]

    dict_json['train_data'] = train_data_cont[0]
    dict_json['folds'] = folds_data
    dict_json['metrics'] = metrics_columns
    dict_json['validate'] = model_data
    with open(path_json,'w') as f:
        json.dump(dict_json, f,sort_keys=True,indent=5)