import importlib
import pandas as pd
import os.path


def str_to_class(module_name, class_name):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            raise AttributeError(f'Class does not exist: {class_name}')
    except ImportError:
        raise ImportError(f'Module does not exist: {module_name}')
    return class_ or None


def csv_to_h5(fname):
    df = pd.read_csv(fname)
    df.rename(columns={"acoustic_data": "s", "time_to_failure": "ttf"}, inplace=True)
    new_fname = os.path.splitext(fname)[0] + ".h5"
    df.to_hdf(new_fname, key='table')