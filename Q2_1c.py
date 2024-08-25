# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the path to the directory of the current script
combined_dir = os.path.join(base_dir, "Combined")

# Load the dataset using np.genfromtxt
X_train = np.genfromtxt(os.path.join(combined_dir, 'X_train.txt'), delimiter=',', dtype=np.float64)
y_train = np.genfromtxt(os.path.join(combined_dir, 'y_train.txt'), delimiter=',', dtype=np.float64)
X_test = np.genfromtxt(os.path.join(combined_dir, 'X_test.txt'), delimiter=',', dtype=np.float64)
y_test = np.genfromtxt(os.path.join(combined_dir, 'y_test.txt'), delimiter=',', dtype=np.float64)

# Combine the datasets
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Split the data into training and testing sets with a different seed value
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

# Reshape X_train and X_test to 2D arrays for training
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the decision tree classifier
model = DecisionTreeClassifier(random_state=seed)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy, precision, recall, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
