import importlib
from modulefinder import ModuleFinder

def fold_to_str(fold_class):
    if 'cv' in fold_class.__dict__:
        return ''.join([str(fold_class.__class__).split('.')[-1], str({k:l for k, l in fold_class.__dict__.items() if l is not None and k is not 'cv'})])
    else:
        return str(fold_class)



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