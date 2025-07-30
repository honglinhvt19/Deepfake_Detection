import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(model, dataset):
    y_true = []
    y_pred = []
    
    for frames, labels in dataset:
        predictions = model.predict(frames, verbose=0)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
    }
    
    return metrics