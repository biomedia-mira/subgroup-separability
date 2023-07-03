from typing import Dict

import numpy as np
import sklearn.metrics as skm


def compute_overall_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute relevant classification metrics for binary classification tasks, where
    1 is the positive label and 0 is the negative label.

    N.B. Do not use in multi-class classification tasks.

    Args:
        scores (np.ndarray): Array of predicted probability scores, shape (N,).
            This is usually the output after a Sigmoid function.
        labels (np.ndarray): Integer array of labels, shape (N,).
        threshold (float, optional): Threshold for converting scores to predictions.
            Defaults to 0.5.

    Returns:
        Dict[str, float]: Dictionary of scalar metrics. Includes:
        - accuracy
        - auc (area under the ROC curve)
    """
    preds = (scores > threshold).astype(int)
    assert preds.shape == labels.shape, f"{preds.shape} != {labels.shape}"
    target_names = ["negative_class", "positive_class"]
    report = skm.classification_report(
        y_true=labels, y_pred=preds, output_dict=True, target_names=target_names
    )

    return {
        "accuracy": report["accuracy"],  # type: ignore
        "auc": skm.roc_auc_score(y_true=labels, y_score=scores),  # type: ignore
    }


def compute_fairness_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    attributes: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute relevant fairness metrics for 2-subgroup binary classification tasks,
    where 1 is the positive label and 0 is the negative label.

    N.B. Do not use in multi-class classification tasks or in tasks with more than 2
    subgroups.

    Args:
        scores (np.ndarray): Array of predicted probability scores, shape (N,).
            This is usually the output after a Sigmoid function.
        labels (np.ndarray): Integer array of labels, shape (N,).
        attributes (np.ndarray): Integer array of subgroup labels, shape (N,).
        threshold (float, optional): Threshold for converting scores to predictions.
            Defaults to 0.5.

    Returns:
        Dict[str, float]: Dictionary of scalar metrics. Includes:
        - accuracy for each subgroup
        - disease prevalence for each subgroup
        - prevalence of each subgroup within dataset
    """
    preds = (scores > threshold).astype(int)
    assert (
        preds.shape == labels.shape == attributes.shape
    ), f"{preds.shape} != {labels.shape} != {attributes.shape}"
    target_names = ["negative_class", "positive_class"]

    group_0_mask = attributes == 0

    report_group_0 = skm.classification_report(
        y_pred=preds[group_0_mask],
        y_true=labels[group_0_mask],
        output_dict=True,
        target_names=target_names,
    )
    report_group_1 = skm.classification_report(
        y_pred=preds[~group_0_mask],
        y_true=labels[~group_0_mask],
        output_dict=True,
        target_names=target_names,
    )

    num_group_0 = np.sum(attributes == 0)
    prevalence_group_0 = num_group_0 / len(attributes)
    num_group_1 = np.sum(attributes == 1)
    prevalence_group_1 = num_group_1 / len(attributes)

    disease_prevalence_group_0 = np.sum(labels[group_0_mask]) / num_group_0
    disease_prevalence_group_1 = np.sum(labels[~group_0_mask]) / num_group_1

    accuracy_group_0 = report_group_0["accuracy"]  # type: ignore
    accuracy_group_1 = report_group_1["accuracy"]  # type: ignore

    return {
        "accuracy_group_0": accuracy_group_0,
        "accuracy_group_1": accuracy_group_1,
        "disease_prevalence_group_0": disease_prevalence_group_0,
        "disease_prevalence_group_1": disease_prevalence_group_1,
        "prevalence_group_0": prevalence_group_0,
        "prevalence_group_1": prevalence_group_1,
    }  # type: ignore
