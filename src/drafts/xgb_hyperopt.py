from folds.folds import CustomFold
from hyperopt import fmin, hp, tpe, STATUS_OK
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import sys
sys.path.append('../data_processing/')
sys.path.append('../../')
sys.path.append('../../src/')


def score(params):
    print("Training with params: ")
    print(params)

    np.random.seed(0)
    folds = CustomFold(n_splits=9, shuffle=True, fragmentation=0, pad=150)
    loss_list = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data)):
        X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]
        y_train, y_valid = y_train_data.iloc[train_index], y_train_data.iloc[valid_index]

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X_train)
        X_train = pd.DataFrame(preprocessor.transform(X_train))
        X_valid = pd.DataFrame(preprocessor.transform(X_valid))

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        loss_list.append(mean_absolute_error(predictions, y_valid))

    loss = np.mean(loss_list)
    return {'loss': loss, 'status': STATUS_OK}


def optimize(random_state=314159265):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """

    space = {
        'n_estimators': hp.choice('n_estimators', np.arange(20, 500, 10, dtype=int)),
        'eta': hp.quniform('eta', 0.005, 0.5, 0.025),
        'max_depth': hp.choice('max_depth', np.arange(1, 6, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 6, dtype=int)),
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
                max_evals=250)
    return best


if __name__ == '__main__':

    df = pd.read_hdf('../../data/e3.h5', key='table')

    train_data = df.drop(['ttf'], axis=1)
    y_train_data = df['ttf']

    best_hyperparams = optimize()
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)


