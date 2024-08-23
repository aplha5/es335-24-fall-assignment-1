from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class Node:
    feature: int = None
    threshold: float = None
    left: 'Node' = None
    right: 'Node' = None
    value: float = None

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index","mse"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(X) < self.min_samples_split:
            return self._create_leaf(y)
        
        best_split = self._get_best_split(X, y)
        if best_split is None:
            return self._create_leaf(y)

        left_data, right_data = split_data(X, y, best_split["feature"], best_split["threshold"])
        left_node = self._build_tree(left_data["X_left"], left_data["y_left"], depth + 1)
        right_node = self._build_tree(right_data["X_right"], right_data["y_right"], depth + 1)
        
        return {"feature": best_split["feature"], "threshold": best_split["threshold"],
                "left": left_node, "right": right_node}

    def _get_best_split(self, X, y):
        num_features = X.shape[1]
        features = X.columns

        best_feature, best_threshold = opt_split_attribute(X, y, self.criterion, features)
        if best_feature is None:
            return None

        return {"feature": best_feature, "threshold": best_threshold}

    def _calculate_split_score(self, y, left_y, right_y):
        return information_gain(y, left_y, right_y, self.criterion)

    def _create_leaf(self, y: pd.Series):
      if check_ifreal(y):
        # For regression, return the mean of the values
        return y.mean()
      else:
        # For classification, return the most common class
        return y.mode()[0]


    def predict(self, X):
        return X.apply(self._predict_sample, axis=1)

    def _predict_sample(self, sample):
        node = self.root
        while isinstance(node, dict):
            if sample[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node

    def plot(self):
        # A simple way to visualize your tree (you can make this more complex)
        self._plot_tree(self.root)

    def _plot_tree(self, node, depth=0):
        if not isinstance(node, dict):
            print("\t" * depth, node)
        else:
            print("\t" * depth, f"[Feature {node['feature']} <= {node['threshold']}]")
            self._plot_tree(node["left"], depth + 1)
            self._plot_tree(node["right"], depth + 1)

