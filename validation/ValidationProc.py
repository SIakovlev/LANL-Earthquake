import pandas as pd
import numpy as np
from folds import str_to_models


"""
ValidationModule

Date: 19.02.2019

Description:

"""

class ValidationBase:

    def __init__(self, **kwargs):
        pass

    def create_model(self,**kwargs):
       self.model = str_to_models(kwargs['model'])
       self.model= self.model(**kwargs['model_params'])

    def train_model(self, path, folds, **kwargs):

        raise NotImplementedError

        pass

    def summary(self, obj, path, key='table', **kwargs):

        raise NotImplementedError

        obj.to_hdf(path, key)



class ValidationSklearn(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationSklearn, self).__init__(**kwargs)
        self.create_model(**kwargs)

    def train_model(self, path, folds, **kwargs):

        raise NotImplementedError

        pass

    def summary(self, obj, path, key='table', **kwargs):

        raise NotImplementedError

        obj.to_hdf(path, key)

class ValidationXgb(ValidationBase):
    def __init__(self, **kwargs):
        super(ValidationXgb, self).__init__(**kwargs)
        self.create_model(**kwargs)

    def train_model(self, path, folds, **kwargs):

        raise NotImplementedError

        pass

    def summary(self, obj, path, key='table', **kwargs):

        raise NotImplementedError

        obj.to_hdf(path, key)
