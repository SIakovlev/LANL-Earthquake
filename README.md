# LANL-Earthquake
LANL Earthquake Prediction Challenge: https://www.kaggle.com/c/LANL-Earthquake-Prediction#description

Notes:
- Fold: `CustomFold(n_splits=9, shuffle=True, fragmentation=0, pad=150)`
- 10 runs is equivalent to 90 model trainings

| Features config | Model           | Params  | 10 runs | 100 runs | 300 runs | Public score |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|-----:|
| [e1](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e1.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.8, 'eta': 0.07, 'eval_metric': 'mae', 'gamma': 0.6, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 1.0, 'tree_method': 'gpu_hist'} | **2.011** | **2.092**| - | 1.650 |
| [e3](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e3.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.7, 'eta': 0.15, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 4, 'min_child_weight': 4, 'n_estimators': 21, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.8, 'tree_method': 'gpu_hist'}  | **2.031** | **2.11** | **2.1556** | - |
| [e6](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e6.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.95, 'eta': 0.015, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 6, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.95, 'tree_method': 'gpu_hist'} | **2.013** | **2.094** | **2.1418** | 1.680 |
