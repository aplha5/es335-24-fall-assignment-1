# Library imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

# Constants
time = 10
offset = 100
folders = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
classes = {"WALKING": 1, "WALKING_UPSTAIRS": 2, "WALKING_DOWNSTAIRS": 3, "SITTING": 4, "STANDING": 5, "LAYING": 6}

base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the path to the directory of the current script
dataset_dir = os.path.join(base_dir, "ACTIVITIES")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Data Preparation
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X = []
y = []
for folder in folders:
    files = os.listdir(os.path.join(dataset_dir, folder))
    for file in files:
        df = pd.read_csv(os.path.join(dataset_dir, folder, file), sep=",", header=0)
        df = df[offset:offset + time * 50]
        X.append(df.values)
        y.append(classes[folder])

X = np.array(X)
y = np.array(y)
print("Dataset shape: ", X.shape)

# Split the dataset into training and testing sets
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Decision Tree Training
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Train the decision tree classifier
model = DecisionTreeClassifier(random_state=seed)
model.fit(X_train, y_train)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Few-Shot Prompting
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Select a few examples from the training data (one per activity)
few_shot_examples = []
few_shot_labels = []

for label in np.unique(y_train):
    example_idx = np.where(y_train == label)[0][0]
    few_shot_examples.append(X_train[example_idx])
    few_shot_labels.append(y_train[example_idx])

few_shot_examples = np.array(few_shot_examples)
few_shot_labels = np.array(few_shot_labels)

# Combine few-shot examples with the test data for prediction
X_few_shot = np.concatenate((few_shot_examples, X_test))
y_few_shot = np.concatenate((few_shot_labels, y_test))

# Predict using the decision tree model
y_pred_few_shot = model.predict(X_few_shot)

# Evaluate the model's performance on the new test query
accuracy_few_shot = accuracy_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):])
precision_few_shot = precision_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):], average='macro')
recall_few_shot = recall_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):], average='macro')
conf_matrix_few_shot = confusion_matrix(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):])

# Output the results
print(f"Few-Shot Prompting Accuracy: {accuracy_few_shot:.4f}")
print(f"Few-Shot Prompting Precision: {precision_few_shot:.4f}")
print(f"Few-Shot Prompting Recall: {recall_few_shot:.4f}")
print("Few-Shot Prompting Confusion Matrix:")
print(conf_matrix_few_shot)
