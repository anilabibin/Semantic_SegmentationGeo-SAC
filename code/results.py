
import os
import json

import numpy as np
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    num_classes,
    ignore_index=None
):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Ground truth and predictions must have equal length."
        )

    valid = (
        (y_true >= 0) &
        (y_true < num_classes) &
        (y_pred >= 0) &
        (y_pred < num_classes)
    )

    if ignore_index is not None:
        valid &= (y_true != ignore_index)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError("No valid labels for metric calculation.")

    labels = np.arange(num_classes)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    tp = np.diag(cm).astype(float)

    fp = cm.sum(axis=0) - tp

    fn = cm.sum(axis=1) - tp

    # Per-class metrics
    precision = tp / np.maximum(tp + fp, 1e-12)

    recall = tp / np.maximum(tp + fn, 1e-12)

    f1 = (
        2 * precision * recall /
        np.maximum(precision + recall, 1e-12)
    )

    iou = (
        tp /
        np.maximum(tp + fp + fn, 1e-12)
    )

    # Classes absent from ground truth are excluded
    # from mean class metrics.
    present = cm.sum(axis=1) > 0

    oa = tp.sum() / max(cm.sum(), 1)

    macc = recall[present].mean()

    miou = iou[present].mean()

    mean_precision = precision[present].mean()

    mean_recall = recall[present].mean()

    mean_f1 = f1[present].mean()

    return {
        "OA": float(oa),
        "mAcc": float(macc),
        "mIoU": float(miou),
        "Precision": float(mean_precision),
        "Recall": float(mean_recall),
        "F1": float(mean_f1),
        "Per_class_IoU": iou.tolist(),
        "Per_class_Precision": precision.tolist(),
        "Per_class_Recall": recall.tolist(),
        "Per_class_F1": f1.tolist(),
        "Confusion_Matrix": cm.tolist()
    }


# --------------------------------------------------
# 2. PRINT METRICS
# --------------------------------------------------

def print_metrics(metrics):

    print("\n" + "=" * 45)

    print("SEMANTIC SEGMENTATION PERFORMANCE")

    print("=" * 45)

    for key in [
        "OA",
        "mAcc",
        "mIoU",
        "Precision",
        "Recall",
        "F1"
    ]:

        print(
            f"{key:15s}: {metrics[key] * 100:.2f}%"
        )

    print("\nPer-class IoU:")

    for index, value in enumerate(
        metrics["Per_class_IoU"]
    ):

        print(
            f"Class {index}: {value * 100:.2f}%"
        )

    print("\nConfusion Matrix:")

    print(
        np.asarray(metrics["Confusion_Matrix"])
    )

    print("=" * 45)


# --------------------------------------------------
# 3. SAVE METRICS
# --------------------------------------------------

def save_metrics(metrics, output_path):

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    with open(output_path, "w") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("Metrics saved:", output_path)
