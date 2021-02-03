import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ---------------- metric functions for numpy -----------------------
def type_acc_np(preds, labels, **kwargs):
    """ Type accuracy ratio  """
    seq_mask = kwargs['seq_mask']

    type_pred = preds['types'][seq_mask]
    type_label = labels['types'][seq_mask]

    return np.mean(type_pred == type_label)


def time_rmse_np(preds, labels, **kwargs):
    """ RMSE for time predictions """
    seq_mask = kwargs['seq_mask']
    dt_pred = preds['dtimes'][seq_mask]
    dt_label = labels['dtimes'][seq_mask]

    rmse = np.sqrt(np.mean((dt_pred - dt_label) ** 2))
    return rmse


def time_mae_np(preds, labels, **kwargs):
    """ MAE for time predictions """
    seq_mask = kwargs['seq_mask']
    dt_pred = preds['dtimes'][seq_mask]
    dt_label = labels['dtimes'][seq_mask]

    dt_pred = np.reshape(dt_pred, [-1])
    dt_label = np.reshape(dt_label, [-1])

    mae = np.mean(np.abs(dt_pred - dt_label))
    return mae


def marks_rmse_np(preds, labels, **kwargs):
    """ RMSE for marks  """
    seq_mask = kwargs['seq_mask']
    pred = preds['marks'][seq_mask]
    label = labels['marks'][seq_mask]

    # pred negative value mask
    pred[pred < 0] = 0

    rmse = np.sqrt(np.mean((pred - label) ** 2))
    return rmse


def mape_np(preds, labels, **kwargs):
    """
    MAPE ratio, one can use thresh=0.01 to mask the zeros
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    threshold = kwargs['threshold']

    # zero mask
    mask = labels > threshold
    preds = preds[mask]
    labels = labels[mask]

    mape = np.mean(np.abs(preds - labels) / labels)
    return mape


def auc_np(predict_prob, labels):
    """ AUC ratio """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    return roc_auc


def precision_recall_curve_np(predict_prob, labels):
    """ Precision - Recall curve  """
    precision, recall, thresholds = precision_recall_curve(labels, predict_prob)
    return [precision, recall]


def get_metric_functions(metric_name_list):
    """ Get metric functions from a list of metric name    """
    metric_functions = []
    for metric_name in metric_name_list:
        metric_functions.append(eval(metric_name + '_np'))
    return metric_functions


def get_metrics_callback_from_names(metric_names):
    """ Metrics function callbacks    """
    metric_functions = get_metric_functions(metric_names)

    def metrics(preds, labels, **kwargs):
        """ call metrics functions """
        res = dict()
        for metric_name, metric_func in zip(metric_names, metric_functions):
            res[metric_name] = metric_func(preds, labels, **kwargs)
        return res

    return metrics
