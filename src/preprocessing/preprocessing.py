class PreprocessingBase:
    """
    Base class for preprocessing. All inherited classes must implement .__call__(df)
    """

    def __init__(self):
        pass

    def fit(self, df):
        """
        Fit preprocessing params to data if needed e.g. normalization: need to calculate mean and variance
        before transform data
        :param df: Pandas Dataframe
        :return:
        """
        raise NotImplementedError

    def transform(self, df):
        """
        Perform preprocessing operation
        :param df: Pandas Dataframe
        :return:
        """
        raise NotImplementedError


