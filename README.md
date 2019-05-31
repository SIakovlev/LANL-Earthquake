# LANL-Earthquake
LANL Earthquake Prediction Challenge: https://www.kaggle.com/c/LANL-Earthquake-Prediction#description

# Team <a name="team"></a>

|      Name  |          Email |    Responsibilities |
|:------------:|:------------:|:-----------:|
| Denish Shitov | - | - |
| Alexey Shaymanov | - | - |
| Pavel Tolmachev | - | - |
| Sergey Iakovlev | siakovlev@student.unimelb.edu.au |  - |

Notes:
- Fold: `CustomFold(n_splits=9, shuffle=True, fragmentation=0, pad=150)`
- 10 runs is equivalent to 90 model trainings

Best performing models:

| Feature config | Model           | Params  | 10 runs score(std) | 100 runs score(std) | 300 runs score(std) | Public score |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|-----:|
| [e7](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e7.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 1.00, 'eta': 0.165, 'eval_metric': 'mae', 'gamma': 0.95, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 0.60, 'tree_method': 'gpu_hist'} | **2.0096** |-| **2.1368** (0.7929) | 1.646 |
| [e7](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e9.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.59, 'eta': 0.273, 'eval_metric': 'mae', 'gamma': 0.82, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 0.78, 'tree_method': 'gpu_hist'} | **2.0105** | **2.0917** (0.7735)| **2.1393** (0.7966) | 1.650 |
| [e1](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e1.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.8, 'eta': 0.07, 'eval_metric': 'mae', 'gamma': 0.6, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 1.0, 'tree_method': 'gpu_hist'} | **2.011** | **2.092**| **2.1397** (0.7959) | 1.650 |
| [e6](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e6.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.95, 'eta': 0.015, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 6, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.95, 'tree_method': 'gpu_hist'} | **2.013** | **2.094** | **2.1418** | 1.680 |
| [e3](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e3.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.7, 'eta': 0.15, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 4, 'min_child_weight': 4, 'n_estimators': 21, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.8, 'tree_method': 'gpu_hist'}  | **2.031** | **2.1099** | **2.1556** (0.7787) | - |

Other models:


| Feature config | Model           | Params  | 10 runs score(std) | 100 runs score(std) | 300 runs score(std) | Public score |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|-----:|
[e6](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e6.json) | AdaBoost |{'learning_rate': 0.23, 'loss': 'linear', 'n_estimators': 13, 'random_state': 0} | **2.0728** | - |- | - |
