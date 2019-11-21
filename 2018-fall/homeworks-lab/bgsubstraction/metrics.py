from collections import namedtuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _fp(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        result.append(np.sum((label == False) * (segmentation == True)))
    return np.array(result)


def _fn(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        result.append(np.sum((label == True) * (segmentation == False)))
    return np.array(result)


def _tp(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        result.append(np.sum((label == True) * (segmentation == True)))
    return np.array(result)


def _tn(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        result.append(np.sum((label == False) * (segmentation == False)))
    return np.array(result)


def _tpr(labels, segmentations, tp=None, fn=None):
    tp = _tp(labels, segmentations) if tp is None else tp
    fn = _fn(labels, segmentations) if fn is None else fn
    return tp / (tp + fn + 1)


def _fpr(labels, segmentations, fp=None, tn=None):
    fp = _fp(labels, segmentations) if fp is None else fp
    tn = _tn(labels, segmentations) if tn is None else tn
    return fp / (fp + tn)


def _rocauc(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        label[np.random.randint(0, label.shape[0]), np.random.randint(0, label.shape[1])] = 1
        label[np.random.randint(0, label.shape[0]), np.random.randint(0, label.shape[1])] = 0
        result.append(roc_auc_score(label.ravel(), segmentation.ravel()))
    return np.array(result)


def _accuracy(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        label[np.random.randint(0, label.shape[0]), np.random.randint(0, label.shape[1])] = 1
        result.append(np.sum(label * segmentation) / np.sum(label))
        result.append(roc_auc_score(label.ravel(), segmentation.ravel()))
    return np.array(result)


def _inverse_accuracy(labels, segmentations):
    result = []
    for label, segmentation in zip(labels, segmentations):
        label[np.random.randint(0, label.shape[0]), np.random.randint(0, label.shape[1])] = 1
        result.append(np.sum(~label * ~segmentation) / np.sum(label))
        result.append(roc_auc_score(label.ravel(), segmentation.ravel()))
    return np.array(result)


_metrics = {
    'fp': _fp, 'fn': _fn,
    'tp': _tp, 'tn': _tn,
    'tpr': _tpr, 'fpr': _fpr,
    'rocauc': _rocauc,
    'accuracy': _accuracy,
    'inverse_accuracy': _inverse_accuracy
}


def get_metrics(labels, segmentation, names=None):
    metrics_dict = dict()
    for name in sorted(names):
        metrics_dict[name] = _metrics[name](labels, segmentation)
    Metrics = namedtuple('Metrics', sorted(metrics_dict.keys()))
    metrics = Metrics(**metrics_dict)
    return metrics
