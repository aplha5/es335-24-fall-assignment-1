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
folders = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the path to the directory of the current script
combined_dir = os.path.join(base_dir, "Combined") 

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Train Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_train=[]
y_train=[]
dataset_dir = os.path.join(combined_dir,"Train")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))

    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_train.append(df.values)
        y_train.append(classes[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Test Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_test=[]
y_test=[]
dataset_dir = os.path.join(combined_dir,"Test")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))
    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_test.append(df.values)
        y_test.append(classes[folder])

X_test = np.array(X_test)
y_test = np.array(y_test)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Final Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# concatenate the training and testing data
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))

# split the data into training and testing sets. Change the seed value to obtain different random splits.
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Decision Tree Training and Evaluation
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Reshape X_train and X_test to 2D arrays for training
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the decision tree classifier
model = DecisionTreeClassifier(random_state=seed)
model.fit(X_train, y_train)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Few-Shot Prompting
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Select a few examples from the training data (one per activity)
few_shot_examples = []
few_shot_labels = []

for label in np.unique(y_train):
    example_idx = np.where(y_train == label)[0][0]
    few_shot_examples.append(X_train[example_idx])
    few_shot_labels.append(y_train[example_idx])

few_shot_examples = np.array(few_shot_examples)
few_shot_labels = np.array(few_shot_labels)

# Preprocess the few-shot examples similarly
few_shot_examples = few_shot_examples.reshape(few_shot_examples.shape[0], -1)

# Load and preprocess your new data from the "ACTIVITIES" folder
X_new = []
y_new = []
dataset_dir = os.path.join(base_dir, "ACTIVITIES")
for folder in folders:
  files=os.listdir(os.path.join(dataset_dir,folder))
  for file in files:
        df = pd.read_csv(os.path.join(dataset_dir,folder, file), sep=",", header=0)
        df = df[offset:offset + time * 50]
        X_new.append(df.values)
        y_new.append(classes[folder])

X_new = np.array(X_new)
y_new = np.array(y_new)
# Reshape new data for prediction
X_new = X_new.reshape(X_new.shape[0], -1)

# Combine few-shot examples with the test data
X_few_shot = np.concatenate((few_shot_examples, X_new))
y_few_shot = np.concatenate((few_shot_labels, y_new))

# Make predictions on the combined data
y_pred_few_shot = model.predict(X_few_shot)

# Evaluate the model's performance on the new test query
accuracy_few_shot = accuracy_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):])
precision_few_shot = precision_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):], average='macro')
recall_few_shot = recall_score(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):], average='macro')
conf_matrix_few_shot = confusion_matrix(y_few_shot[len(few_shot_labels):], y_pred_few_shot[len(few_shot_labels):])

print(f"Few-Shot New Data Accuracy: {accuracy_few_shot:.4f}")
print(f"Few-Shot New Data Precision: {precision_few_shot:.4f}")
print(f"Few-Shot New Data Recall: {recall_few_shot:.4f}")
print("Few-Shot New Data Confusion Matrix:")
print(conf_matrix_few_shot)
