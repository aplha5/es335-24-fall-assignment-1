import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from tree.utils import *
from metrics import *

np.random.seed(42)

# Reading and cleaning the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep='\s+', header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

data = data[data.horsepower != '?']
data = data.drop(columns=['car name'])
data['horsepower'] = pd.to_numeric(data['horsepower'])

X = data.drop(columns=['mpg'])
y = data['mpg']
X = one_hot_encoding(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom decision tree
custom_tree = DecisionTree(criterion="mse", max_depth=5, min_samples_split=10)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)

# Scikit-learn decision tree
from sklearn.tree import DecisionTreeRegressor
sklearn_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=5, min_samples_split=10, random_state=42)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)

# Evaluation
rmse_custom = rmse(y_pred_custom, y_test)
mae_custom = mae(y_pred_custom, y_test)
rmse_sklearn = rmse(y_pred_sklearn, y_test)
mae_sklearn = mae(y_pred_sklearn, y_test)

print(f"Custom Decision Tree - RMSE: {rmse_custom}, MAE: {mae_custom}")
print(f"Scikit-learn Decision Tree - RMSE: {rmse_sklearn}, MAE: {mae_sklearn}")

# Plotting the custom decision tree
custom_tree.plot()
