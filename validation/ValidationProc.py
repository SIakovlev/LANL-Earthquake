import pandas as pd
import numpy as np
from folds import str_to_models, fold_to_str

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer
from utils import str_to_class
from sklearn.metrics import SCORERS
import pickle
import os

import nn_test

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

    def prepare_data(self,x_data,y_data, t_index, v_index):
        return x_data.iloc[t_index], \
               x_data.iloc[v_index],\
               y_data.iloc[t_index], \
               y_data.iloc[v_index]

    def create_models(self,**kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = str_to_models(m)
            print(m, m_p)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["name"] = str(m)
            self.models_features[i_m]["features"] = m_p
            self.models_features.append({})

    def train_model(self, train_data, y_train_data, folds, metric, path_to_save, folds_param):
            raise NotImplementedError

    def save_summary_of_model(self,name, fold_params, model_params, metric):
        save_temp_dict = {"fold_data": self.folds_data,
                          "fold_features" : fold_params,
                          "metric" : metric,
                          "model": model_params}
        print(save_temp_dict)

        #check if file exist
        while True:
            if os.path.isfile(name+'.pkl') is True:
                name += '_new'
            else:
                name += '.pkl'
                break
        print(name)
        with open(name, "wb") as f:
            pickle.dump(save_temp_dict, f)


class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def check_metrics(self,metrics):
        for metric in metrics:
            if metric not in SCORERS:
                raise AttributeError(f"not metric {metric} for sklearn models")


    def print_models(self):
        for l_item in self.models_list:
            print(l_item)

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save):

        train_data, y_train_data = self.reshape_data(train_data, y_train_data)

        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                X_train, X_valid, y_train, y_valid = self.prepare_data(train_data,y_train_data, train_index, valid_index)
                model.fit(X_train, y_train)
                score_data = get_scorer(metric)(model, X_valid, y_valid)
                self.folds_data.append(score_data)

            self.save_summary_of_model(''.join([path_to_save, str(self.__class__.__name__)]), folds_param, self.models_features[num_model], metric)
            self.folds_data = []


class ValidationMatrix(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationMatrix, self).__init__(**kwargs)
        self.create_models(**kwargs)

    # not work now
    def train_model(self, train_data, y_train_data, folds, metric, path_to_save, folds_param):
        #X_train, X_valid, y_train, y_valid = self.prepare_data(train_data,y_train_data, train_index, valid_index)
        pass


class ValidationCustom(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationCustom, self).__init__(**kwargs)
        self.path =  kwargs['path']
        self.create_models(**kwargs)

    def create_models(self,**kwargs):
        for i_m, (m, m_p) in enumerate(kwargs['model'].items()):
            self.model = str_to_class(self.path, m)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.models_features[i_m]["name"] = str(m)
            self.models_features[i_m]["features"] = m_p
            self.models_features.append({})

    def check_metrics(self,metrics):
        for metric in metrics:
            for m in self.models_list:
                if metric not in m.metrics.keys():
                    raise AttributeError(f"not metric {metric} for model {str(m)}")

    def train_model(self, train_data, y_train_data, folds, folds_param, metric, path_to_save):
        for num_model, m in enumerate(self.models_list):
            try:
                self.folds_data = m.train_model(train_data, y_train_data)
                self.save_summary_of_model(''.join([path_to_save, str(self.__class__.__name__)]), folds_param,
                                           self.models_features[num_model], metric)
                self.folds_data = []
            except AttributeError or NameError:
                raise AttributeError(f'Method train not implement for class : {str(m)}')