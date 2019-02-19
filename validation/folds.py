import importlib
from modulefinder import ModuleFinder



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