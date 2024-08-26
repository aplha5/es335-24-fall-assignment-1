import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

# Step 1: Load the data
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the path to the directory of the current script
combined_dir = os.path.join(base_dir, "Combined")

# Load data using pandas, specifying that there is no header in the files
X_train = pd.read_csv(os.path.join(combined_dir, "X_train.txt"), sep=r"\s+", header=None).values
y_train = pd.read_csv(os.path.join(combined_dir, "y_train.txt"), sep=r"\s+", header=None).values.ravel()  # ravel to flatten the array
X_test = pd.read_csv(os.path.join(combined_dir, "X_test.txt"), sep=r"\s+", header=None).values
y_test = pd.read_csv(os.path.join(combined_dir, "y_test.txt"), sep=r"\s+", header=None).values.ravel()


# Combine train and test sets
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Step 2: Split the data into training and testing sets again
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Import necessary libraries for plotting
import matplotlib.pyplot as plt

# Define the range of depths to test
depths = range(2, 9)
accuracies = []

# Train the Decision Tree for each depth and store the accuracy
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    
    print(f"Depth: {depth}, Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Plot the accuracy vs depth
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.xticks(depths)
plt.grid(True)
plt.show()

