# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:05:53 2025

@author: 22472
"""

import numpy as np
import os
import time

from sklearn.metrics import accuracy_score
from scipy import sparse  
from sklearn.datasets import load_svmlight_file, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set project root directory
PROJECT_ROOT = r"C:\Users\22472\Desktop\审稿意见\logistic_regression_algorithms"

# Add project root to Python path
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import all algorithm classes from algorithm file
from logistic_regression_algorithms import (
    BaseLogisticRegression,
    LqLogisticRegression,
    HALogisticRegression,
    EPDCAeLogisticRegression,
    NEPDCALogisticRegression,
    APGLogisticRegression,
    NL0RLogisticRegression
)

def get_nl0r_lambda(data_name):
    """Get hyperparameters for NL0R algorithm for each dataset"""
    if data_name == 'RCV1':
        return 1e-5
    elif data_name == 'Arcene':
        return 1e-5
    elif data_name == 'Gisette':
        return 1e-5
    elif data_name == 'Leuke':
        return 1e-4
    elif data_name == 'Duke':
        return 2e-3
    elif data_name == 'Wine':
        return 2e-4
    elif data_name == 'Breast_Cancer':
        return 2e-4
    elif data_name.startswith('UCR_'):
        # Parameter settings for UCR time series datasets
        ucr_name = data_name[4:]
        
        if ucr_name == "Coffee":
            return 4e-2
        elif ucr_name == "ECG200":
            return 1e-3
        elif ucr_name == "BirdChicken":
            return 5e-5
        else:
            return 5e-5
    else:
        return 1e-4

def compare_algorithms(X_train, y_train, X_test, y_test, lambda_=0.00002, t=0.0001, nl0r_lambda=None):
    """Compare performance of different algorithms"""
    
    algorithms = {
        'IHT_{1/2}': LqLogisticRegression(lambda_=lambda_*100, verbose=True),
        'NL0R': NL0RLogisticRegression(lambda_init=nl0r_lambda, max_iter=2000),  # Use separate lambda
        'HA': HALogisticRegression(lambda_=lambda_, t=t, verbose=True),
        'EPDCAe': EPDCAeLogisticRegression(lambda_=lambda_, t=t, verbose=True),
        'NEPDCA': NEPDCALogisticRegression(lambda_=lambda_, t=t, verbose=True),
        'APG': APGLogisticRegression(lambda_=lambda_, t=t, verbose=True)
    }
    
    results = {}
    
    for name, model in algorithms.items():
        print(f"\n{'='*50}")
        print(f"Training {name} algorithm...")
        
        # Display used lambda value
        if name == 'NL0R':
            print(f"NL0R using lambda: {nl0r_lambda}")
        else:
            print(f"{name} using lambda: {lambda_}")
            
        start_time = time.perf_counter()
        
        try:
            # Special handling for NL0R algorithm
            if name == 'NL0R':
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start_time
                accuracy = model.score(X_test, y_test)
                
                # Fix: Check if get_nonzero_weights is a method or property
                if hasattr(model.get_nonzero_weights, '__call__'):
                    # If it's a method, need to call it
                    n_nonzero, _ = model.get_nonzero_weights()
                else:
                    # If it's a property, access directly
                    n_nonzero, _ = model.get_nonzero_weights
                    
                sparsity = model.get_sparsity()
                final_loss = model.loss_history_[-1] if hasattr(model, 'loss_history_') and model.loss_history_ else float('inf')
            else:
                # Unchanged handling for other algorithms
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start_time
                accuracy = model.score(X_test, y_test)
                
                # Check if get_nonzero_weights is a method or property
                if hasattr(model.get_nonzero_weights, '__call__'):
                    n_nonzero, _ = model.get_nonzero_weights()
                else:
                    n_nonzero, _ = model.get_nonzero_weights
                    
                sparsity = model.get_sparsity()
                final_loss = model.loss_history[-1] if hasattr(model, 'loss_history') and model.loss_history else float('inf')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'train_time': train_time,
                'n_nonzero': n_nonzero,
                'sparsity': sparsity,
                'final_loss': final_loss
            }
            
            # Choose appropriate display precision based on time magnitude
            if train_time < 0.01:
                time_str = f"{train_time:.6f}s"
            elif train_time < 1:
                time_str = f"{train_time:.4f}s"
            else:
                time_str = f"{train_time:.2f}s"
            
            # Use scientific notation for all loss values
            loss_str = f"{final_loss:.2e}"
                
            print(f"{name} results: Accuracy={accuracy:.4f}, Training time={time_str}, "
                  f"Non-zero weights={n_nonzero}, Sparsity={sparsity:.4f}, Loss={loss_str}")
                  
        except Exception as e:
            print(f"{name} algorithm training failed: {e}")
            import traceback
            traceback.print_exc()  # Print detailed error information
            results[name] = {
                'model': None,
                'accuracy': 0.0,
                'train_time': 0.0,
                'n_nonzero': 0,
                'sparsity': 0.0,
                'final_loss': float('inf')
            }
    
    return results

def load_and_preprocess_dataset(data_path, data_name):
    """Load and preprocess a single dataset"""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {data_name}")
    
    if data_name.startswith('UCR_'):
        # Load UCR time series dataset
        ucr_name = data_name[4:]  # Remove 'UCR_' prefix
        print(f"=== Using UCR Dataset: {ucr_name} for Testing Custom Logistic Regression ===")
        
        try:
            # Use tslearn to load UCR dataset
            from tslearn.datasets import UCR_UEA_datasets
            ucr_datasets_loader = UCR_UEA_datasets()
            X_train_ucr, y_train_ucr, X_test_ucr, y_test_ucr = ucr_datasets_loader.load_dataset(ucr_name)
            
            # Merge training and test sets, then re-split (for unified processing)
            X = np.vstack([X_train_ucr, X_test_ucr])
            y = np.hstack([y_train_ucr, y_test_ucr])
            
            # Flatten time series into feature vectors
            X_flat = X.reshape(X.shape[0], -1)
            
            X = X_flat
            
            # Ensure labels are in 0,1 format
            unique_labels = np.unique(y)
            if len(unique_labels) == 2:
                # If labels are not 0,1, perform mapping
                if not np.array_equal(unique_labels, [0, 1]):
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    y_binary = np.array([label_map[label] for label in y])
                else:
                    y_binary = y
            else:
                print(f"Warning: {ucr_name} is not a binary classification problem, using first two categories")
                # Select first two categories
                mask = (y == unique_labels[0]) | (y == unique_labels[1])
                X = X[mask]
                y = y[mask]
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                y_binary = np.array([label_map[label] for label in y])
            
            print(f"UCR dataset {ucr_name} loaded successfully")
            print(f"Data shape: {X.shape}")
            print(f"Label distribution: 0: {np.sum(y_binary == 0)}, 1: {np.sum(y_binary == 1)}")
            
            # Use original UCR split
            X_train = X_train_ucr.reshape(X_train_ucr.shape[0], -1)
            X_test = X_test_ucr.reshape(X_test_ucr.shape[0], -1)
            
            unique_labels = np.unique(np.hstack([y_train_ucr, y_test_ucr]))
            if len(unique_labels) == 2 and not np.array_equal(unique_labels, [0, 1]):
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                y_train = np.array([label_map[label] for label in y_train_ucr])
                y_test = np.array([label_map[label] for label in y_test_ucr])
            else:
                y_train = y_train_ucr
                y_test = y_test_ucr
                
        except Exception as e:
            print(f"Failed to load UCR dataset {ucr_name}: {e}")
            return None, None, None, None
            
    elif data_path in ['wine', 'breast_cancer']:
        # Load sklearn built-in datasets
        print(f"=== Using {data_name} Dataset for Testing Custom Logistic Regression ===")
        
        if data_name == 'Wine':
            wine = load_wine()
            X, y = wine.data, wine.target
            mask = (y == 0) | (y == 1)
            X = X[mask]
            y = y[mask]
        elif data_name == 'Breast_Cancer':
            cancer = load_breast_cancer()
            X, y = cancer.data, cancer.target
        
        y_binary = y
        print(f"Labels: {np.unique(y_binary)}")
        print(f"Label distribution: 0: {np.sum(y_binary == 0)}, 1: {np.sum(y_binary == 1)}")
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        
    else:
        # Load file datasets
        print(f"=== Using {data_name} Dataset for Testing Custom Logistic Regression ===")
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"Error: Dataset file {data_path} does not exist")
            return None, None, None, None
        
        X, y = load_svmlight_file(data_path)
        print(f"Original labels: {np.unique(y)}")
        y_binary = np.where(y == 1, 1, 0)
        print(f"Transformed labels: {np.unique(y_binary)}")
        print(f"Label distribution: 0: {np.sum(y_binary == 0)}, 1: {np.sum(y_binary == 1)}")
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Data preprocessing
    if X_train.shape[0] <= 10000 and X_train.shape[1] <= 10000:
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        
        scaler = StandardScaler(with_mean=True)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Add bias term
    if sparse.issparse(X_train):
        X_train_with_bias = sparse.hstack([X_train, np.ones((X_train.shape[0], 1))]).tocsr()
        X_test_with_bias = sparse.hstack([X_test, np.ones((X_test.shape[0], 1))]).tocsr()
    else:
        X_train_with_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_test_with_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    print(f"Training set shape after adding bias: {X_train_with_bias.shape}")

    # Ensure y is numpy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train_with_bias, X_test_with_bias, y_train, y_test

def generate_relative_accuracy_plot(accuracy_data, dataset_names, algorithm_order):
    """Generate stacked bar chart of relative accuracy"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set global font and style
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 20,  # Increase axis label size
            'xtick.labelsize': 18,  # Increase x-tick label size
            'ytick.labelsize': 18,  # Increase y-tick label size
            'legend.fontsize': 18,  # Increase legend text size
            'figure.dpi': 300,
            'font.family': 'DejaVu Sans',  # Ensure Chinese font compatibility
            'mathtext.fontset': 'stix'  # Use STIX math font
        })
        
        # Create accuracy DataFrame
        accuracy_df = pd.DataFrame(accuracy_data, index=dataset_names)
        
        # Calculate relative accuracy (divide by max accuracy for each dataset)
        relative_accuracy_df = accuracy_df.copy()
        for dataset in dataset_names:
            max_accuracy = accuracy_df.loc[dataset].max()
            if max_accuracy > 0:  # Avoid division by zero
                relative_accuracy_df.loc[dataset] = accuracy_df.loc[dataset] / max_accuracy
            else:
                relative_accuracy_df.loc[dataset] = 0
        
        # Calculate cumulative relative accuracy
        cumulative_relative_accuracy = relative_accuracy_df.sum(axis=0)
        
        print(f"\n{'='*80}")
        print("Original accuracy data:")
        print(accuracy_df.round(4))
        print("\nRelative accuracy data:")
        print(relative_accuracy_df.round(4))
        print("\nCumulative relative accuracy:")
        for algo in algorithm_order:
            if algo in cumulative_relative_accuracy:
                acc = cumulative_relative_accuracy[algo]
                print(f"{algo}: {acc:.4f}")
        
        # Set figure and axes
        width = 0.6  # Bar width
        fig, ax = plt.subplots(figsize=(16, 10))  # Increase figure size
        
        # Use blue gradient colors
        colors = plt.cm.Blues(np.linspace(0.5, 1, len(dataset_names)))
        
        # Convert algorithm names to LaTeX format
        algorithm_labels = []
        for algo in algorithm_order:
            if algo == 'IHT_{1/2}':
                algorithm_labels.append(r'$\mathrm{IHT}_{1/2}$')  # LaTeX format
            else:
                algorithm_labels.append(algo)
        
        # Create dataset name mapping table (using provided format)
        dataset_name_mapping = {
            'Wine': 'Wine',
            'Breast_Cancer': 'Breast', 
            'UCR_ECG200': 'ECG200',
            'UCR_Coffee': 'Coffee',
            'UCR_BirdChicken': 'BirdC',
            'Gisette': 'Gisette',
            'Leuke': 'Leuke',
            'Duke': 'Duke',
            'Arcene': 'Arcene',
            'RCV1': 'RCV1'
        }
        
        # Create stacked bar chart
        bottom = np.zeros(len(algorithm_order))
        for i, (dataset, color) in enumerate(zip(dataset_names, colors)):
            values = []
            for algo in algorithm_order:
                if algo in relative_accuracy_df.columns and dataset in relative_accuracy_df.index:
                    values.append(relative_accuracy_df.loc[dataset, algo])
                else:
                    values.append(0)
            
            values = np.array(values)
            
            # Use mapped dataset name
            display_name = dataset_name_mapping.get(dataset, dataset)
            
            ax.bar(algorithm_labels, values, width, bottom=bottom, 
                   label=display_name, color=color, edgecolor='white', linewidth=0.5)
            bottom += values
        
        # 修改这里：显示三位小数
        for i, algo_label in enumerate(algorithm_labels):
            total = bottom[i]  # Current bar total
            if total > 0:  # Only display if there is data
                # 修改格式为保留三位小数
                ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom', 
                       fontsize=18, fontweight='bold')  # Increase text size
        
        # Set labels
        ax.set_ylabel('Cumulative ACCRatio', fontsize=20, fontweight='bold')
        #ax.set_xlabel('Algorithms', fontsize=20, fontweight='bold')
        
        # Move legend to outside right column
        legend = ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=18,  # Increase legend text size
                          ncol=1, title='Datasets', title_fontsize=16)
        legend.get_title().set_ha('center')
        
        # Set y-axis range
        y_max = cumulative_relative_accuracy.max() + 1
        plt.ylim(0, y_max)
        
        # Axis settings
        plt.xticks(fontsize=18, rotation=45)  # Increase x-axis algorithm name size
        plt.yticks(fontsize=18)
        
        # Add grid lines
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Remove figure title
        # plt.title('Cumulative Relative Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        plt.savefig('cumulative_relative_accuracy.eps', bbox_inches='tight', dpi=300, format='eps')
        plt.savefig('cumulative_relative_accuracy.png', bbox_inches='tight', dpi=300)
        plt.savefig('cumulative_relative_accuracy.pdf', bbox_inches='tight', dpi=300)
        
        print("\nImage saved as: cumulative_relative_accuracy.eps/.png/.pdf")
        
        # Display graph
        plt.show()
        
        # Performance ranking (using original algorithm names)
        print(f"\n{'='*50}")
        print("Algorithm performance ranking:")
        print(f"{'='*50}")
        ranking = cumulative_relative_accuracy.sort_values(ascending=False)
        for i, (algo, score) in enumerate(ranking.items(), 1):
            print(f"{i}. {algo}: {score:.4f}")
            
    except ImportError as e:
        print(f"\nCannot generate chart, missing required libraries: {e}")
        print("Please install: pip install pandas matplotlib")
    except Exception as e:
        print(f"\nError generating chart: {e}")

def main():
    # Set data directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    # Define datasets in specified order
    datasets = [
        ('wine', 'Wine'),
        ('breast_cancer', 'Breast_Cancer'),
        ('ucr_ECG200', 'UCR_ECG200'),
        ('ucr_Coffee', 'UCR_Coffee'),
        ('ucr_BirdChicken', 'UCR_BirdChicken'),
        (os.path.join(data_dir, 'gisette.binary'), 'Gisette'),
        (os.path.join(data_dir, 'leuke.binary'), 'Leuke'),
        (os.path.join(data_dir, 'duke.binary'), 'Duke'),
        (os.path.join(data_dir, 'arcene.binary'), 'Arcene'),
        (os.path.join(data_dir, 'rcv1_train.binary'), 'RCV1')
    ]

    # Overall results for all datasets
    all_results = {}

    # Iterate through all datasets
    for data_path, data_name in datasets:
        print(f"\n{'#'*100}")
        print(f"{'#'*45} {data_name} {'#'*45}")
        print(f"{'#'*100}")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset(data_path, data_name)
        
        if X_train is None:
            print(f"Skipping dataset {data_name}, loading failed")
            continue

        # Set regularization parameters
        if data_name == 'RCV1':
            lambda_ = 1e-6
            t = 1e-4
        elif data_name == 'Arcene':
            lambda_ = 1e-6
            t = 1e-4
        elif data_name == 'Gisette':
            lambda_ = 1e-6
            t = 1e-4
        elif data_name == 'Leuke':
            lambda_ = 2e-5
            t = 1e-4
        elif data_name == 'Duke':
            lambda_ = 3e-5
            t = 1e-4
        elif data_name == 'Wine':
            lambda_ = 1e-5
            t = 1e-4
        elif data_name == 'Breast_Cancer':
            lambda_ = 1e-5
            t = 1e-4
        elif data_name.startswith('UCR_'):
            # Parameter settings for UCR time series datasets
            ucr_name = data_name[4:]  # Remove 'UCR_' prefix
            
            # Set different parameters based on UCR dataset
            if ucr_name == "Coffee":
                lambda_ = 4e-5
                t = 1e-4
            elif ucr_name == "ECG200":
                lambda_ = 1e-5
                t = 1e-4
            elif ucr_name == "BirdChicken":
                lambda_ = 5e-6
                t = 1e-4
            else:
                # Default parameters (for other unlisted UCR datasets)
                lambda_ = 5e-6
                t = 1e-4

        # Get separate lambda parameter for NL0R algorithm
        nl0r_lambda = get_nl0r_lambda(data_name)
        
        print(f"\nUsing parameters: lambda={lambda_}, NL0R_lambda={nl0r_lambda}, t={t}")

        # Compare all algorithms
        results = compare_algorithms(X_train, y_train, X_test, y_test, 
                                   lambda_=lambda_, t=t, nl0r_lambda=nl0r_lambda)
        
        # Save results for this dataset
        all_results[data_name] = results
        
        # Output summary for this dataset
        print(f"\n{'='*80}")
        print(f"{data_name} dataset algorithm comparison summary:")
        print(f"{'Algorithm':<12} {'Accuracy':<8} {'Training Time':<15} {'Non-zero':<10} {'Sparsity':<8} {'Final Loss':<12}")
        for name, result in results.items():
            # Choose appropriate display precision based on time magnitude
            if result['train_time'] < 0.01:
                time_str = f"{result['train_time']:.6f}s"
            elif result['train_time'] < 1:
                time_str = f"{result['train_time']:.4f}s"
            else:
                time_str = f"{result['train_time']:.2f}s"
            
            # Use scientific notation for all loss values
            loss_str = f"{result['final_loss']:.2e}"
                
            print(f"{name:<12} {result['accuracy']:.4f}   {time_str:<15} "
                  f"{result['n_nonzero']:<10} {result['sparsity']:.4f}    {loss_str}")

        # Find best algorithm
        best_algorithm = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest algorithm: {best_algorithm[0]}, Accuracy: {best_algorithm[1]['accuracy']:.4f}")

    # Output table results for all datasets at once
    print(f"\n{'#'*150}")
    print(f"{'#'*60} Complete Results Table for All Datasets {'#'*60}")
    print(f"{'#'*150}")
    
    # Define algorithm order
    algorithm_order = ['IHT_{1/2}', 'NL0R', 'HA', 'EPDCAe', 'NEPDCA', 'APG']
    
    # 1. Accuracy table
    print(f"\n{'='*120}")
    print(f"{'Accuracy Comparison':^120}")
    print(f"{'='*120}")
    print(f"{'Dataset':<15} ", end="")
    for algo in algorithm_order:
        print(f"{algo:<12} ", end="")
    print()
    print("-" * 120)
    
    # Store accuracy data for subsequent calculations
    accuracy_data = {}
    dataset_names = []
    
    for data_name, results in all_results.items():
        dataset_names.append(data_name)
        print(f"{data_name:<15} ", end="")
        for algo in algorithm_order:
            if algo in results:
                accuracy = results[algo]['accuracy']
                print(f"{accuracy:.4f}     ", end="")
                
                # Store accuracy data
                if algo not in accuracy_data:
                    accuracy_data[algo] = []
                accuracy_data[algo].append(accuracy)
            else:
                print(f"{'N/A':<12} ", end="")
                if algo not in accuracy_data:
                    accuracy_data[algo] = []
                accuracy_data[algo].append(0.0)  # Fill missing values with 0
        print()
    
    # Calculate relative accuracy and generate stacked bar chart
    if accuracy_data and dataset_names:
        generate_relative_accuracy_plot(accuracy_data, dataset_names, algorithm_order)

    # 2. Training time table
    print(f"\n{'='*120}")
    print(f"{'Training Time Comparison (seconds)':^120}")
    print(f"{'='*120}")
    print(f"{'Dataset':<15} ", end="")
    for algo in algorithm_order:
        print(f"{algo:<12} ", end="")
    print()
    print("-" * 120)
    
    for data_name, results in all_results.items():
        print(f"{data_name:<15} ", end="")
        for algo in algorithm_order:
            if algo in results:
                time_val = results[algo]['train_time']
                if time_val < 0.01:
                    time_str = f"{time_val:.6f}"
                elif time_val < 1:
                    time_str = f"{time_val:.4f}"
                else:
                    time_str = f"{time_val:.2f}"
                print(f"{time_str:<12} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()
    
    # 3. Sparsity table
    print(f"\n{'='*120}")
    print(f"{'Sparsity Comparison':^120}")
    print(f"{'='*120}")
    print(f"{'Dataset':<15} ", end="")
    for algo in algorithm_order:
        print(f"{algo:<12} ", end="")
    print()
    print("-" * 120)
    
    for data_name, results in all_results.items():
        print(f"{data_name:<15} ", end="")
        for algo in algorithm_order:
            if algo in results:
                sparsity = results[algo]['sparsity']
                print(f"{sparsity:.4f}     ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()
    
    # 4. Final loss table
    print(f"\n{'='*120}")
    print(f"{'Final Loss Comparison':^120}")
    print(f"{'='*120}")
    print(f"{'Dataset':<15} ", end="")
    for algo in algorithm_order:
        print(f"{algo:<12} ", end="")
    print()
    print("-" * 120)
    
    for data_name, results in all_results.items():
        print(f"{data_name:<15} ", end="")
        for algo in algorithm_order:
            if algo in results:
                final_loss = results[algo]['final_loss']
                if final_loss == float('inf'):
                    loss_str = "Inf"
                else:
                    loss_str = f"{final_loss:.2e}"
                print(f"{loss_str:<12} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()


if __name__ == "__main__":
    main()