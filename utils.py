import tensorflow as tf
import numpy as np


def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculate intersection over union for the given batch."""
    # Cast both arrays to boolean arrays according to a cut-off of 0.5
    y_pred = (y_pred > 0.5)
    y_true = (y_true > 0.5)

    # Mark each pixel as True if both predicts True (TP)
    intersection_mask = tf.math.logical_and(y_pred, y_true)

    # Mark each pixel as True if either true positive, false negative, or false
    # positive (TP + FP + FN)
    union_mask = tf.math.logical_or(y_pred, y_true)

    # Calculate sum of true positives
    intersection = tf.math.count_nonzero(intersection_mask, axis=(1, 2))

    # Calculate sum of true positives fales negatives/positives
    union = tf.math.count_nonzero(union_mask, axis=(1, 2))

    # Set union to 1 where there are no true positives in order to prevent
    # division by zero
    dividing_union = tf.where(union == 0, tf.ones_like(union), union)

    # Calculate intersection over union
    ious = intersection / dividing_union

    # Set IoU to 1 where there are no true positives
    ious = tf.where(union == 0, tf.ones_like(ious), ious)

    # Return the mean IoU across the batch
    return tf.reduce_mean(ious)



def get_metrics(model, X_test, y_test):

    area = np.sum(y_test, axis=(1,2,3))
    nonempty = area > 0

    model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics + [
                    "binary_accuracy",
                    "FalseNegatives",
                    "FalsePositives",
                    "Precision",
                    "Recall",
                    iou
                ],)

    evaluation = model.evaluate(X_test[nonempty], y_test[nonempty])
    metrics = {
        name: value
        for name, value
        in zip(model.metrics_names, evaluation)
                    }
    return metrics
