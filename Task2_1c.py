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


seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')        
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)
