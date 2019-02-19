import pandas as pd
import numpy as np
from folds import str_to_models

from sklearn import metrics as sklearn_metrics
from sklearn.metrics import get_scorer

"""
ValidationModule

Date: 19.02.2019

Description:

"""

class ValidationBase:

    def __init__(self, **kwargs):
        self.models_list = []
        self.metrics = kwargs['metrics']
        # self.summary = pd.DataFrame()

    def create_models(self,**kwargs):
        for models_name, m_parameters in kwargs["model_params"].items():
            self.model = str_to_models(models_name)
            self.model = self.model(**m_parameters)
            self.models_list.append(self.model)


    def train_model(self, train_data, y_train_data, folds):
        raise NotImplementedError

    def summary(self, path, key='table'):
        # self.summary.to_hdf(path, key)
        pass


class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.create_models(**kwargs)

    def print_models(self):
        for l in self.models_list:
            print(self.model)

    def train_model(self, train_data, y_train_data, folds):
        #DIMENSION OF DATA MUST BE MORE THAN 1
        if train_data.values.ndim == 1:
            train_data = train_data.values.reshape(-1,1)
            y_train_data = y_train_data.values.reshape(-1,1)

        for model in self.models_list:
            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
                X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
                y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]
                for metric in self.metrics:
                    model.fit(X_train, y_train)
                    score_data = get_scorer(metric)(model, X_valid, y_valid)
                    print(f' model = {model.get_params()} | fold = {folds.__class__.__name__} | metric : {metric} = {score_data}')


class ValidationMatrix(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationMatrix, self).__init__(**kwargs)
        self.create_model(**kwargs)

    # not work now
    def train_model(self, train_data, y_train_data, folds):
        train_data = self.model.DMatrix(data=train_data, label=y_train_data)


