import importlib
import re


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


def natural_keys(text):
    """
    Keys for sorting (for MemoryManagement module)
    :param text: path to a file
    :type text: str
    :return: list ['part', number]
    :rtype: list
    """
    # grab the last name from the path and remove .h5 extention
    filename = re.sub('.h5$', '', text.split('/')[-1])
    return [atoi(c) for c in re.split(r'(\d+)', filename)]


def atoi(text):
    return int(text) if text.isdigit() else text
