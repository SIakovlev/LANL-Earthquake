from src.utils import str_to_class

def check_metrics(metrics):
    metrics_class = {}
    try:
        for m in metrics:
            metrics_class[m] = str_to_class('sklearn.metrics', m)
    except AttributeError:
        raise AttributeError(f'Metric does not exist: {m}')
    return metrics_class