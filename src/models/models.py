class ModelBase:
    """
    Base classes for all the custom models
    """
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

