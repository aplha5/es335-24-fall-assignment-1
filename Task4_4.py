import pandas as pd
from langchain_groq.chat_models import ChatGroq
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import re
import time
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
Groq_Token = os.getenv("GROQ_TOKEN")

if Groq_Token is None:
    raise ValueError("API key not found in the .env file.")

groq_models = {
    "llama3-70b": "llama3-70b-8192", 
    "mixtral": "mixtral-8x7b-32768", 
    "gemma-7b": "gemma-7b-it",
    "llama3.1-70b": "llama-3.1-70b-versatile",
    "llama3-8b": "llama3-8b-8192",
    "llama3.1-8b": "llama-3.1-8b-instant",
    "gemma-9b": "gemma2-9b-it"
}

folders = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]
classes = {"WALKING": 1, "WALKING_UPSTAIRS": 2, "WALKING_DOWNSTAIRS": 3, "SITTING": 4, "STANDING": 5, "LAYING": 6}

base_dir = os.path.dirname(os.path.abspath(__file__))
activities_dir = os.path.join(base_dir, "ACTIVITIES")


def preprocess_data(file_path, offset=100, time=10):
    df = pd.read_csv(file_path, sep=",", header=0)
    df = df[offset:offset + time * 50]
    return df.values.flatten()

def load_data_from_folder(folder_path, offset=100, time=10):
    X = []
    y = []
    for folder in folders:
        folder_path_full = os.path.join(folder_path, folder)
        files = os.listdir(folder_path_full)
        for file in files:
            file_path = os.path.join(folder_path_full, file)
            X.append(preprocess_data(file_path, offset, time))
            y.append(classes[folder])
    return X, y

X, y = load_data_from_folder(activities_dir)
X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split
seed = 4
X_train, X_new, y_train, y_new = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Define example data for Few-Shot Prompting
def create_few_shot_prompt(activity_examples, data):
    example_prompts = "\n".join([f"Activity Data: {ex}\nActivity: {label}" for ex, label in activity_examples])
    query = f"""
* You are a model for activity recognition based on sensor data.
* Your task is to classify the activity from the given data into one of the predefined categories.
* Provide the activity label based on the data provided.

Here are a few examples:
{example_prompts}

Activity Data: {data}
"""
    return query

# Create Few-Shot prompts
activity_examples = [(X_train[i], y_train[i]) for i in range(min(5, len(X_train)))]  # Use a subset of training data for examples

# Collect predictions with retry mechanism
def get_predictions_with_retry(X_new, activity_examples, retries=3, delay=5):
    y_pred = []
    for data in X_new:
        prompt = create_few_shot_prompt(activity_examples, data)
        
        for attempt in range(retries):
            try:
                model_name = "llama3-70b"  # Choose appropriate model
                llm = ChatGroq(model=groq_models[model_name], api_key=Groq_Token, temperature=0)
                answer = llm.invoke(prompt)
                response_text = answer.content.strip()
                match = re.search(r'Activity: (\d+)', response_text)
                if match:
                    predicted_class = int(match.group(1))
                    y_pred.append(predicted_class)
                    break  # Exit retry loop if successful
                else:
                    y_pred.append(-1)  # Append a default value for invalid predictions
                    break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait before retrying
                else:
                    y_pred.append(-1)  # Handle final failure case
    
    return np.array(y_pred)


y_pred = get_predictions_with_retry(X_new, activity_examples)

if len(y_pred) == 0:
    print("No predictions were made.")
else:
    accuracy = accuracy_score(y_new, y_pred)
    precision = precision_score(y_new, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_new, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_new, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
