import pandas as pd
import numpy as np
from folds import *

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer
from sklearn.metrics import SCORERS
import pickle
import os
import xgboost as xgb

import importlib.util
spec = importlib.util.spec_from_file_location("utils", os.path.join(os.getcwd(), '../src/utils.py'))
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

"""
ValidationModule

Date: 19.02.2019

Description:

"""


class ValidationBase:

    def __init__(self, **kwargs):
        self.score_data = []
        self.models_list = []
        self.models_features = [{}]

    def reshape_data(self,train_data, y_train_data):
        # DIMENSION OF DATA MUST BE MORE THAN 1
        if train_data.values.ndim == 1:
            train_data = train_data.values.reshape(-1, 1)
            y_train_data = y_train_data.values.reshape(-1, 1)
        return (train_data, y_train_data)

    def print_models(self):
        for l_item in self.models_list:
            print(l_item)

    def create_model(self, model):
        raise NotImplementedError

    def create_models(self,**kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = self.create_model(m)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["name"] = str(m)
            self.models_features[i_m]["features"] = m_p
            self.models_features.append({})

    def train_models(self, train_data, y_train_data, folds, folds_param, metric, path_to_save,metric_classes):

        train_data, y_train_data = self.reshape_data(train_data, y_train_data)

        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):

                X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
                y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]

                model.fit(X_train, y_train)
                y_predict = model.predict(X_valid)
                score_d = metric_classes(y_predict, y_valid)

                self.score_data.append(score_d)

            self.save_summary_of_model(path_to_save, [folds_param, self.models_features[num_model], {'metric':metric}])
            self.score_data = []

    def save_summary_of_model(self, name, *list_data):

        dfObj = pd.DataFrame()
        for l in list_data[0]:
            for d_k in l.keys():
                dfObj[d_k] = [l[d_k]]

        dfObj["score_data"] = [self.score_data]

        if not os.path.exists(name):
            with open(name, 'wb') as f:
                pickle.dump(dfObj,f)
        else:
            with open(name, 'rb') as f:
                modDfObj = pickle.load(f)

            with open(name, '+wb') as f:
                concat = pd.concat([dfObj, modDfObj])
                pickle.dump(concat,f)
                print(concat)
                del concat,modDfObj

class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def create_model(self, model):
        return str_to_models(model)



class ValidationCustom(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationCustom, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def create_model(self,m):
        return find_and_load_class(m)
