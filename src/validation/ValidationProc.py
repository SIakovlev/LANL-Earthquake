import pandas as pd
import numpy as np
from folds import *

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer
from sklearn.metrics import SCORERS
import pickle
import os
import xgboost as xgb

from src.validation.nn_test import CustomNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


from src.utils import str_to_class

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
        model =  str_to_class(__name__, model)
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

        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                # get data
                X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
                y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]
                #clean model
                model = self._create_model(self.models_features[num_model]["model_name"], self.models_features[num_model]["model_params"])
                # train model
                model.fit(X_train, y_train)
                # validate
                y_predict = model.predict(X_valid)
                # compute all scores
                for metric_name, metric_value in metric_classes.items():
                    if fold_n==0:
                        self.score_data[metric_name] = []

                    score_d = metric_value(y_predict, y_valid)
                    self.score_data[metric_name].append(score_d)
            # save summary
            params_to_save.append(self.models_features[num_model])
            self._save_summary_of_model(path_to_save, params_to_save)
            self.score_data = {}

    def _save_summary_of_model(self, path, *args):

        '''
        Save summary for all models in validator instance
        :param path:
        :param kwargs: columns in summary
        '''

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
                dfObj[d] = None


        if not os.path.exists(path):
            with open(path, 'wb') as f:
                pickle.dump(dfObj,f)
        else:
            with open(path, 'rb') as f:
                modDfObj = pickle.load(f)

            with open(path, '+wb') as f:
                concat = pd.concat([dfObj, modDfObj])
                pickle.dump(concat,f)
                print(concat)
                del concat,modDfObj


