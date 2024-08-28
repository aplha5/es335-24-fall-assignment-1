import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from joblib import Parallel, delayed

np.random.seed(42)
num_average_time = 10  

def generate_fake_data(N, M, case='classification'):
    """Function to generate fake data for testing."""
    X = np.random.randint(0, 2, size=(N, M))  # Binary features
    if case == 'classification':
        y  = np.random.randint(0, 2, size=N)  # Binary target for classification
    else:
        y = np.random.rand(N)  # Continuous target for regression
    return pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(M)]), pd.Series(y)

def measure_runtime(X, y, max_depth, criterion, case):
    """Function to measure the runtime of fitting and predicting with the decision tree."""
    dt = DecisionTree(criterion=criterion, max_depth=max_depth)
    
    # Measure fitting time
    start_time = time.time()
    dt.fit(X, y)
    fit_time = time.time() - start_time

    # Measure prediction time
    test_data = X.sample(frac=0.2, random_state=42)
    start_time = time.time()
    y_pred = dt.predict(test_data)
    predict_time = time.time() - start_time
    
    return fit_time, predict_time

def run_experiment_for_params(N, M, depth, criterion, case):
    X, y = generate_fake_data(N, M, case)
    fit_time, predict_time = measure_runtime(X, y, depth, criterion, case)
    return N, M, fit_time, predict_time, case, criterion, depth

def run_experiments():
    """Run experiments and plot results."""
    depths = [3, 5, 7]
    sample_sizes = [100, 500, 1000]
    feature_counts = [10, 50, 100]
    cases = ['classification', 'regression']
    criteria = ['information_gain', 'gini_index', 'mse']

    results = Parallel(n_jobs=-1)(delayed(run_experiment_for_params)(N, M, depth, criterion, case) 
                                  for case in cases 
                                  for criterion in criteria 
                                  for depth in depths 
                                  for N in sample_sizes 
                                  for M in feature_counts)

    results_df = pd.DataFrame(results, columns=['N', 'M', 'Fit Time', 'Predict Time', 'Case', 'Criterion', 'Depth'])

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    for case in cases:
        df_case = results_df[results_df['Case'] == case]
        
        for criterion in criteria:
            df_criterion = df_case[df_case['Criterion'] == criterion]
            
            axs[0].plot(df_criterion['N'], df_criterion['Fit Time'], label=f'{case} - {criterion}')
            axs[1].plot(df_criterion['N'], df_criterion['Predict Time'], label=f'{case} - {criterion}')
        
        axs[0].set_title(f'Fit Time vs. Sample Size for {case}')
        axs[0].set_xlabel('Number of Samples (N)')
        axs[0].set_ylabel('Fit Time (seconds)')
        axs[0].legend()
        
        axs[1].set_title(f'Predict Time vs. Sample Size for {case}')
        axs[1].set_xlabel('Number of Samples (N)')
        axs[1].set_ylabel('Predict Time (seconds)')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiments()
