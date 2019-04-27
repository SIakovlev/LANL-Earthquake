import pandas as pd


def summarize(scores=None,
              train_data_fname=None,
              preproc=None,
              folds=None,
              model=None, **kwargs):
    columns = ["data_fname",
               "preproc_name",
               "preproc_params",
               "folds_name",
               "folds_params",
               "model_name",
               "model_params"]
    columns.extend(scores.keys())

    row = []
    if train_data_fname is None:
        raise KeyError
    else:
        row.append(train_data_fname)

    # preproc is optional, so put None if missing
    if preproc is not None:
        row.append(preproc['name'])
        del preproc['name']
        row.append(preproc)
    else:
        row.extend([None, None])

    row.append(folds['name'])
    del folds['name']
    row.append(folds)

    row.append(model['name'])
    del model['name']
    row.append(model)

    for k in scores.keys():
        row.append(scores[k])
    df = pd.DataFrame([row], columns=columns)

    return df
