import pandas as pd
import numpy as np
from folds import str_to_models, fold_to_str

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer

"""
ValidationModule

Date: 19.02.2019

Description:

"""

class ValidationBase:

    def __init__(self, **kwargs):
        self.metrics = kwargs['metrics']
        self.summary_model = {}

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
        raise NotImplementedError

    def train_model(self, train_data, y_train_data, folds):
        raise NotImplementedError


    def create_summary(self, fold_type=None):
        for m in self.metrics:
            if fold_type is not None:
                self.summary_model[''.join([m, fold_to_str(fold_type)])] = []
            else:
                self.summary_model[m] = []

    def save_summary_to_h5(self,name, key ='table'):
        pd.DataFrame.from_dict(self.summary_model).to_hdf(name, key)


class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.models_list = []
        self.summ_list = []
        self.create_models(**kwargs)

    def create_models(self,**kwargs):
        for m, m_p in kwargs['model'].items():
            self.model = str_to_models(m)
            self.model = self.model(**m_p)
            self.models_list.append(self.model)
            self.summ_list.append(str(m)+str(m_p))

    def print_models(self):
        for l_item in self.models_list:
            print(l_item)

    def train_model(self, train_data, y_train_data, folds, path):

        train_data, y_train_data = self.reshape_data(train_data, y_train_data)
        self.create_summary(folds)

        for num_model, model in enumerate(self.models_list):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                X_train, X_valid, y_train, y_valid = self.prepare_data(train_data,y_train_data, train_index, valid_index)
                for metric in self.metrics:
                    model.fit(X_train, y_train)
                    score_data = get_scorer(metric)(model, X_valid, y_valid)
                    self.summary_model[metric+fold_to_str(folds)].append(score_data)

            self.save_summary_to_h5(''.join([path, str(self.summ_list[num_model]), '.h5']))
            self.summary_model.fromkeys(self.summary_model, [])

class ValidationMatrix(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationMatrix, self).__init__(**kwargs)
        self.create_model(**kwargs)

    # not work now
    def train_model(self, train_data, y_train_data, folds):
        #X_train, X_valid, y_train, y_valid = self.prepare_data(train_data,y_train_data, train_index, valid_index)
        pass


class ValidationCustom(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationCustom, self).__init__(**kwargs)
        self.create_model(**kwargs)

    def create_model(self,**kwargs):
        pass
    # not work now
    def train_model(self, train_data, y_train_data, folds, path):
        pass