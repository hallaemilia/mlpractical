import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
import os
# import sys
# sys.path.append('/path/to/mlpractical')

from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, CustomActivationLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

plt.style.use('ggplot')

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for num_epochs epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('Error')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    ax_2.set_xlabel('Accuracy')

    grad_plot, grad_ax = optimiser.plot_grad_flow()

    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2, grad_plot, grad_ax

def save_stats(stats, model_run):

    # Extract data from the LaTeX table and save as CSV
    table_data = {
        'Model': ['Baseline'] + ['Dropout']*4 + ['L1 penalty']*4 + ['L2 penalty']*4 + ['Label smoothing'],
        'Hyperparameter_Value': ['-', 0.6, 0.7, 0.85, 0.97, 5e-4, 1e-3, 5e-3, 5e-2, 5e-4, 1e-3, 5e-3, 5e-2, 0.1],
        'Validation_Accuracy': [0.837, 0.807, 0.841, 0.851, 0.854, 0.795, 0.733, 0.0241, 0.0220, 0.851, 0.849, 0.813, 0.392, 0.834],
        'Train_Error': [0.241, 0.549, 0.348, 0.329, 0.244, 0.642, 0.883, 3.850, 3.850, 0.306, 0.356, 0.586, 2.258, 0.837],
        'Validation_Error': [0.533, 0.593, 0.464, 0.434, 0.457, 0.658, 0.894, 3.850, 3.850, 0.460, 0.453, 0.607, 2.256, 1.230]
    }

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Create data directory if it doesn't exist
    data_dir = '/Users/hallaei/UoE/mlp/mlpractical/data'
    os.makedirs(data_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(data_dir, 'regularization_experiments.csv')
    df.to_csv(csv_path, index=False)

    print(f"Data saved to: {csv_path}")
    print("\nDataFrame preview:")
    print(df)