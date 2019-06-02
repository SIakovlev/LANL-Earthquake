# LANL-Earthquake
LANL Earthquake Prediction Challenge: https://www.kaggle.com/c/LANL-Earthquake-Prediction#description

Date: 01 Aug 2018 </pre>

# Table of contents

1. [Team](#team) 
2. [Abstract](#abstract)
3. [Introduction](#intro)
   1. [Project structure](#struct)
   2. [Setup](#setup)
4. [Pipeline architecture](#architecture)
   1. [Data processing](#dp)
   2. [Validation](#cv)
   3. [Testing](#testing)
5. [Results](#results)
6. [Summary](#summary)


# Team <a name="team"></a>

|      Name  |          Email |    Responsibilities |
|:------------:|:------------:|:-----------:|
| Denish Shitov | - | - |
| Alexey Shaymanov | - | - |
| Pavel Tolmachev | - | - |
| Sergey Iakovlev | siakovlev@student.unimelb.edu.au |  - |

# Abstract <a name="abstract"></a>

# Introduction <a name="intro"></a>

## Project structure <a name="struct"></a>

Under the `/src` directory there is the following structure:
* `/configs` - configs for data processing, model training and validation.
* `/data_processing`
  * `dp_run.py` - main data processing script with parameters specified in `dp_config.json`.
  * `feature.py` - `Feature` class implementation.
  * `dp_utils.py` - 
* `/folds` - directory with custom fold implementation
* `/models` - custom models that follow `model.py` interface
* `/preproc` - ?
* `/test_directory` - the main script generating test prediction results
* `/validation` - directory with training and validation scripts:
  * `train_single_model.py` - script for training of a single model with parameters specified in `/configs/train_config.json`
  * `validation_run.py`- script for training of a single/multiple models with parameters specified in `/configs/validation_config.json`

## Configs
Data processing, model training and generation of test results can be managed via the following configs (in `/configs` dir):

* [`dp_config.json`](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config.json)
   - A user needs to specify directories with the original dataframe (`data_dir`), output directory for the processed dataframe (`data_processed_dir`), and their names (`data_fname` and `data_processed_fname`);
   - There are two global parameters: `window_size` and `window_stride` that are used by default during each feature calculation. Note, these parameters can be overriden by each feature locally (see below).
   - `features` is the list of features to be calculated. In the configuration file, each feature has 3 parameters: 
      - `name` - feature name;
      - `on` - the feature caluculation can be enabled or disabled;
      - `functions` - a dictionary of functions with corresponding parameters from `feature.py`.
   An example of a single feature is provided below:
   ```
   {
      "name": "q_05_std_rolling_50",
      "on": true,
      "functions": {
        "r_std": {
          "window_size": 50,
          "window_stride": null
        },
        "w_quantile": {
          "q": 0.05
        }
      }
    }
   ```
* [`train_config.json`]() or [`validation_config.json`]()
* [`multi_test_config.json`](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/multi_test_config.json)

## Setup <a name="setup"></a>

### Windows
### Mac OS
### Linux

# Results <a name="results"></a>

Notes:
- Fold: `CustomFold(n_splits=1, shuffle=True, fragmentation=0, pad=150)`
- 10 runs is equivalent to 90 model trainings

Best performing models:

{'booster': 'gbtree', 'colsample_bytree': 1.0, 'eta': 0.177, 'eval_metric': 'mae', 'gamma': 0.93, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 65001, 'silent': 1, 'subsample': 0.65, 'tree_method': 'gpu_hist'}

| Feature config | Model           | Params  | 10 runs score(std) | 100 runs score(std) | 300 runs score(std) | Public score |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|-----:|
| [e9](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e9.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 1.0, 'eta': 0.177, 'eval_metric': 'mae', 'gamma': 0.93, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 65001, 'silent': 1, 'subsample': 0.65, 'tree_method': 'gpu_hist'} | **2.0092** |-|  |  |
| [e9](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e9.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 1.00, 'eta': 0.165, 'eval_metric': 'mae', 'gamma': 0.95, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 0.60, 'tree_method': 'gpu_hist'} | **2.0096** |-| **2.1368** (0.7929) | 1.646 |
| [e7](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e7.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.59, 'eta': 0.273, 'eval_metric': 'mae', 'gamma': 0.82, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 0.78, 'tree_method': 'gpu_hist'} | **2.0105** | **2.0917** (0.7735)| **2.1393** (0.7966) | 1.650 |
| [e1](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e1.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.8, 'eta': 0.07, 'eval_metric': 'mae', 'gamma': 0.6, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 0, 'silent': 1, 'subsample': 1.0, 'tree_method': 'gpu_hist'} | **2.011** | **2.092**| **2.1397** (0.7959) | 1.650 |
| [e6](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e6.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.95, 'eta': 0.015, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 6, 'min_child_weight': 10, 'n_estimators': 20, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.95, 'tree_method': 'gpu_hist'} | **2.013** | **2.094** | **2.1418** | 1.680 |
| [e3](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e3.json) | XGBRegressor | {'booster': 'gbtree', 'colsample_bytree': 0.7, 'eta': 0.15, 'eval_metric': 'mae', 'gamma': 0.75, 'max_depth': 4, 'min_child_weight': 4, 'n_estimators': 21, 'objective': 'gpu:reg:linear', 'seed': 314159265, 'silent': 1, 'subsample': 0.8, 'tree_method': 'gpu_hist'}  | **2.031** | **2.1099** | **2.1556** (0.7787) | - |

Other models:


| Feature config | Model           | Params  | 10 runs score(std) | 100 runs score(std) | 300 runs score(std) | Public score |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|-----:|
[e6](https://github.com/SIakovlev/LANL-Earthquake/blob/develop/src/configs/dp_config_e6.json) | AdaBoost |{'learning_rate': 0.23, 'loss': 'linear', 'n_estimators': 13, 'random_state': 0} | **2.0728** | - |- | - |
