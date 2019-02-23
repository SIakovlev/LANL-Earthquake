import pandas as pd
import numpy as np
from folds import *

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer
from sklearn.metrics import SCORERS
import pickle
import os

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
        self.folds_data = []
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

    def prepare_data(self,x_data,y_data, t_index, v_index):
        return x_data.iloc[t_index], \
               x_data.iloc[v_index],\
               y_data.iloc[t_index], \
               y_data.iloc[v_index]

    def create_models(self,**kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = str_to_models(m)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["name"] = str(m)
            self.models_features[i_m]["features"] = m_p
            self.models_features.append({})

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save, metric_clases):
            raise NotImplementedError

    def save_summary_of_model(self,name, fold_params, model_params, metric):
        # save_temp_dict = {"fold_data": self.folds_data,
        #                   "fold_features" : fold_params,
        #                   "metric" : metric,
        #                   "model": model_params}
        dfObj = pd.DataFrame()
        dfObj['fold_data'] = self.folds_data,
        dfObj['fold_name'] = fold_params['fold_name'],
        dfObj["fold_params"] = [fold_params['fold_param']],
        dfObj["metric"] = [metric]
        dfObj["model_name"] = [model_params['name']]
        dfObj["model_features"] = [model_params['features']]
        if not os.path.exists(name):
            with open(name, 'wb') as f:
                pickle.dump(dfObj,f)
        else:
            with open(name, 'rb') as f:
                modDfObj = pickle.load(f)

            with open(name, '+wb') as f:
                concat = pd.concat([dfObj, modDfObj])
                pickle.dump(concat,f)
                del concat,modDfObj

class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save,metric_clases):

        train_data, y_train_data = self.reshape_data(train_data, y_train_data)

        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                X_train, X_valid, y_train, y_valid = self.prepare_data(train_data,y_train_data, train_index, valid_index)
                model.fit(X_train, y_train)
                score_data = metric_clases(X_valid, y_valid)
                self.folds_data.append(score_data)

            self.save_summary_of_model(path_to_save, folds_param, self.models_features[num_model], metric)
            self.folds_data = []


class ValidationBoost(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationBoost, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save, metric_clases):
        train_data, y_train_data = self.reshape_data(train_data, y_train_data)
        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                X_train, X_valid, y_train, y_valid = self.prepare_data(train_data, y_train_data, train_index,
                                                                       valid_index)
                model.fit(X_train, y_train)
                score_data = metric_clases(y_valid, model.predict(X_valid))
                self.folds_data.append(score_data)

            self.save_summary_of_model(path_to_save, folds_param,
                                       self.models_features[num_model], metric)
            self.folds_data = []


class ValidationCustom(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationCustom, self).__init__(**kwargs)
        self.path =  kwargs['path']
        self.create_models(**kwargs)

    def create_models(self,**kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = find_and_load_class(m)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["name"] = str(m)
            self.models_features[i_m]["features"] = m_p
            self.models_features.append({})

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save,metric_clases):
        for num_model, m in enumerate(self.models_list):
            try:
                for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                    X_train, X_valid, y_train, y_valid = self.prepare_data(train_data, y_train_data, train_index,
                                                                           valid_index)
                    m.train_model(X_train, y_train)

                    y_probe = m.compute_loss(X_valid, y_valid)
                    score_data = metric_clases(y_valid, y_probe)

                    self.folds_data.append(score_data)


                self.save_summary_of_model(path_to_save, folds_param,
                                           self.models_features[num_model], metric)
                self.folds_data = []
            except AttributeError or NameError:
                raise AttributeError(f'Method train not implement for class : {str(m)}')