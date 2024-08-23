from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy.
    Accuracy is the ratio of correctly predicted samples to the total number of samples.
    """
    # Ensure y_hat and y have the same size
    assert y_hat.size == y.size, "Size of predicted and actual labels must be the same."

    # Ensure y_hat and y are not empty
    assert y_hat.size > 0, "Input series must not be empty."

    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    accuracy_value = correct_predictions / total_predictions
    return accuracy_value


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision.
    Precision is the ratio of true positives to the sum of true positives and false positives.
    """
    # Ensure y_hat and y have the same size
    assert y_hat.size == y.size, "Size of predicted and actual labels must be the same."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    predicted_positives = (y_hat == cls).sum()

    # Avoid division by zero
    if predicted_positives == 0:
        return 0.0

    precision_value = true_positives / predicted_positives
    return precision_value


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall.
    Recall is the ratio of true positives to the sum of true positives and false negatives.
    """
    # Ensure y_hat and y have the same size
    assert y_hat.size == y.size, "Size of predicted and actual labels must be the same."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()

    # Avoid division by zero
    if actual_positives == 0:
        return 0.0

    recall_value = true_positives / actual_positives
    return recall_value


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (rmse).
    RMSE is the square root of the average of squared differences between predicted and actual values.
    """
    # Ensure y_hat and y have the same size
    assert y_hat.size == y.size, "Size of predicted and actual labels must be the same."

    squared_errors = (y_hat - y) ** 2
    mse = squared_errors.mean()
    rmse_value = np.sqrt(mse)
    return rmse_value


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (mae).
    MAE is the average of absolute differences between predicted and actual values.
    """
    # Ensure y_hat and y have the same size
    assert y_hat.size == y.size, "Size of predicted and actual labels must be the same."

    absolute_errors = (y_hat - y).abs()
    mae_value = absolute_errors.mean()
    return mae_value
