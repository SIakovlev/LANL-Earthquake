from folds.folds import CustomFold
from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


def score(params, n=10):
    print(f"Training with params: {params}")

    np.random.seed(0)
    folds = CustomFold(n_splits=9, shuffle=True, fragmentation=0, pad=150)
    loss_list = []
    for _ in range(n):
        for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
            X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
            y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]

            preprocessor = preprocessing.StandardScaler()
            preprocessor.fit(X_train)
            X_train = pd.DataFrame(preprocessor.transform(X_train))
            X_valid = pd.DataFrame(preprocessor.transform(X_valid))

            model = XGBRegressor(**params)
            model.fit(X_train.values, y_train.values)
            predictions = model.predict(X_valid.values)
            loss_list.append(mean_absolute_error(predictions, y_valid))

    loss = np.mean(loss_list)
    print(f"Score: {loss}")
    return {'loss': loss, 'status': STATUS_OK}


def optimize(random_state=314159265):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'eta': hp.quniform('eta', 0.005, 0.3, 0.005),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 8, 1)),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'mae',
        'objective': 'gpu:reg:linear',
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'silent': 1,
        'seed': random_state
    }

    best = fmin(score,
                space,
                algo=tpe.suggest,
                max_evals=2000)
    return best


if __name__ == '__main__':

    df = pd.read_hdf('../../data/e3.h5', key='table')
    train_data = df.drop(['ttf'], axis=1)
    y_train_data = df['ttf']

    best_hyperparams = optimize()
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)


