import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

# def create_sequences(config, X, y, n_step):
#     seq_len = config.seq_len
    
#     sequences = []
#     targets = []
    
#     for i in range(len(X) - seq_len - n_step + 1):
#         seq = X[i:i + seq_len]
#         target = y[i + seq_len + n_step - 1:i + seq_len + n_step]
#         sequences.append(seq)
#         targets.append(target)
#     return torch.stack(sequences), torch.stack(targets)

def create_sequences(config, X, y, input_steps, target_step=1):
    """
    Create input-output pairs for n-step input and k-step output using matrix operations.
    
    Args:
        data (np.ndarray): The original time-series data (1D array).
        input_steps (int): Number of past time steps to use as input.
        target_step (int): Number of steps ahead to predict.
    
    Returns:
        X (torch.Tensor): Input data of shape (num_samples, features=1, input_steps).
        y (torch.Tensor): Target data of shape (num_samples,).
    """
    num_samples = len(X) - input_steps - target_step + 1
    indices = np.arange(num_samples)[:, None] + np.arange(input_steps + target_step)
    X_new = X[indices[:, :input_steps]]
    y_new = y[indices[:, input_steps + target_step - 1]]
    return X_new, y_new

def create_sequences_iter(config, X, y, input_steps, target_step=1):
    """
    Create input-output pairs for n-step input and k-step output using matrix operations.
    
    Args:
        data (np.ndarray): The original time-series data (1D array).
        input_steps (int): Number of past time steps to use as input.
        target_step (int): Number of steps ahead to predict.
    
    Returns:
        X (torch.Tensor): Input data of shape (num_samples, features=1, input_steps).
        y (torch.Tensor): Target data of shape (num_samples,).
    """
    num_samples = len(X) - input_steps - target_step + 1
    indices = np.arange(num_samples)[:, None] + np.arange(input_steps + target_step)
    X_new = X[indices[:, :input_steps]]
    y_new = y[indices[:, input_steps + target_step - 1]]
    return X_new, y_new

def create_sequences_mlp(config, X, y, input_steps, target_step):
    """
    Create input-output pairs for n-step input and k-step output using matrix operations.
    
    Args:
        data (np.ndarray): The original time-series data (1D array).
        input_steps (int): Number of past time steps to use as input.
        target_step (int): Number of steps ahead to predict.
    
    Returns:
        X (torch.Tensor): Input data of shape (num_samples, input_steps).
        y (torch.Tensor): Target data of shape (num_samples,).
    """
    num_samples = len(X) - input_steps - target_step + 1
    indices = np.arange(num_samples)[:, None] + np.arange(input_steps + target_step)
    X = X[indices[:, :input_steps]]
    y = y[indices[:, input_steps + target_step - 1]]
    return X, y

def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def normalize_data(X_train, y_train, X_test, y_test):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train, y_train, X_test, y_test, X_scaler, y_scaler

def unnormalize_predictions(y_pred, y_scaler):
    return y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()