import importlib

import os,sys
import glob

import importlib.util
spec = importlib.util.spec_from_file_location("utils", os.path.join(os.getcwd(), '../src/utils.py'))
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

def check_metrics(metrics):
    metrics_class = []
    try:
        for m in metrics:
            metrics_class.append(foo.str_to_class('sklearn.metrics', m))
    except AttributeError:
        raise AttributeError(f'Metric does not exist: {m}')
    return metrics_class


def str_to_models(class_name):
    with open('models_heads.py') as f:
        for line in f:
            text_data = line.rstrip().replace(' ', '').split("from")[1].split("import")
            if class_name in text_data:
                module_ = importlib.import_module(text_data[0])
                try:
                    class_ = getattr(module_, class_name)
                except AttributeError:
                    raise AttributeError(f'Class does not exist: {class_name}')
                return class_ or None

def find_and_load_class(class_name):
    for file in glob.glob("**/*.py",recursive=True):
        name = os.path.splitext(os.path.basename(file))[0]
        # add package prefix to name, if required
        module_ = __import__(name)
        for member in dir(module_):
            if member  == class_name:
                class_ = foo.str_to_class(module_.__name__, class_name)
                return class_
    raise AttributeError(f'Class with model realization not include in directory: {class_name}')