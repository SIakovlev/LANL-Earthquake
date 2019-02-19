import importlib
from modulefinder import ModuleFinder


def str_to_fold_object(class_name):
    try:
        module_ = importlib.import_module('sklearn.model_selection')
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            raise AttributeError(f'Class does not exist: {class_name}')
    except ImportError:
        raise ImportError(f'Module does not exist: sklearn.model_selection ')
    return class_ or None

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