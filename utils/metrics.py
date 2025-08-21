import tensorflow as tf
import numpy as np
from keras.metrics import AUC, Precision, Recall

def get_metrics(metric_names):
    metric_objs = []
    for m in metric_names:
        m = m.lower()
        if m == "accuracy":
            metric_objs.append("accuracy")
        elif m == "precision":
            metric_objs.append(Precision(name="precision"))
        elif m == "recall":
            metric_objs.append(Recall(name="recall"))
        elif m == "f1":
            metric_objs.append(F1Score(name="f1"))
        elif m == "roc_auc":
            metric_objs.append(AUC(name="roc_auc"))
        elif m == "pr_auc":
            metric_objs.append(AUC(name="pr_auc", curve="PR"))
        else:
            raise ValueError(f"Unknown metric: {m}")
    return metric_objs
