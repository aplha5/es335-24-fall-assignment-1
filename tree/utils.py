import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if isinstance(y, pd.Categorical):
        return False  # Categorical data is considered discrete
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probabilities = Y.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probabilities = Y.value_counts(normalize=True)
    return 1 - np.sum(probabilities ** 2)

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return np.mean((Y - np.mean(Y)) ** 2)

def information_gain(parent_y: pd.Series, left_y: pd.Series, right_y: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using the criterion (entropy, gini index, or MSE)
    """
    if criterion == "information_gain":
        parent_entropy = entropy(parent_y)
        left_entropy = entropy(left_y)
        right_entropy = entropy(right_y)
        weighted_avg = len(left_y) / len(parent_y) * left_entropy + len(right_y) / len(parent_y) * right_entropy
        return parent_entropy - weighted_avg
    elif criterion == "gini_index":
        parent_gini = gini_index(parent_y)
        left_gini = gini_index(left_y)
        right_gini = gini_index(right_y)
        weighted_avg = len(left_y) / len(parent_y) * left_gini + len(right_y) / len(parent_y) * right_gini
        return parent_gini - weighted_avg
    else:  # For regression, using MSE
        parent_mse = mse(parent_y)
        left_mse = mse(left_y)
        right_mse = mse(right_y)
        weighted_avg = len(left_y) / len(parent_y) * left_mse + len(right_y) / len(parent_y) * right_mse
        return parent_mse - weighted_avg

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split upon.
    """
    best_feature = None
    best_threshold = None
    max_gain = -float('inf')

    for feature in features:
        feature_values = X[feature]

        # Convert categorical data to numeric if needed
        if feature_values.dtype.name == 'category':
            feature_values = feature_values.cat.codes
            unique_values = feature_values.unique()
        else:
            unique_values = np.unique(feature_values)
        
        for threshold in unique_values:
            if feature_values.dtype.name == 'category':
                # For categorical data, use equality check
                left_mask = feature_values == threshold
                right_mask = feature_values != threshold
            else:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) > 0 and len(y_right) > 0:
                gain = information_gain(y, y_left, y_right, criterion)
                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature
                    best_threshold = threshold

    return best_feature, best_threshold



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    """
    if X[attribute].dtype.name == 'category':
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value
    else:
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    return {"X_left": X_left, "y_left": y_left}, {"X_right": X_right, "y_right": y_right}

