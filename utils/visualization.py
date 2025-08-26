import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vẽ loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Vẽ accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_frames(frames, title="Video Frames"):
    num_frames = frames.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 3, 3))
    for i in range(num_frames):
        axes[i].imshow(frames[i].astype(np.uint8))
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_eval_curves(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    # ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}", color="blue")
    plt.plot([0,1],[0,1],'--',color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # PR
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred_labels, labels=("Real","Fake")):
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def print_classification_report(y_true, y_pred_labels, target_names=("Real","Fake")):
    report = classification_report(y_true, y_pred_labels, target_names=target_names, digits=4)
    print("=== Classification Report ===")
    print(report)

