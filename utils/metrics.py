import tensorflow as tf
import numpy as np
from keras.metrics import AUC, Precision, Recall

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


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
        elif m == "roc_auc":
            metric_objs.append(AUC(name="roc_auc"))
        elif m == "pr_auc":
            metric_objs.append(AUC(name="pr_auc", curve="PR"))
        else:
            raise ValueError(f"Unknown metric: {m}")
    return metric_objs
