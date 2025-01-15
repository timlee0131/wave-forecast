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

def create_sequences_geometric(config, X, y, look_back, horizon):
    """
    Creates overlapping windows for multivariate time series data.

    Parameters:
        data (torch.Tensor): Input tensor of shape (a, c, b), where
                             a = number of time steps,
                             c = number of observations,
                             b = feature dimension.
        look_back (int): Number of past time steps to use as input.
        horizon (int): Number of future time steps to predict.

    Returns:
        X (torch.Tensor): Input tensor of shape (n, c, look_back, b), where
                          n = number of valid windows.
        y (torch.Tensor): Target tensor of shape (n, c, horizon, b).
    """
    # Get dimensions
    num_samples, num_observations, num_features = X.shape
    
    # Determine the number of valid windows
    n_windows = num_samples - look_back - horizon + 1
    if n_windows <= 0:
        raise ValueError("Not enough data to create windows with the given look_back and horizon sizes.")
    
    # Create input (X) and target (y) windows
    X_new = []
    y_new = []
    
    for i in range(n_windows):
        # Extract look-back window for all observations
        X_window = X[i:i + look_back]  # Shape: (look_back, c, b)
        
        # Extract prediction horizon for all observations
        # y_window = y[i + look_back:i + look_back + horizon]  # Shape: (horizon, c, b)
        y_window = y[i+look_back+horizon-1:i + look_back + horizon]  # Shape: (horizon, c, b)
        
        X_new.append(X_window)
        y_new.append(y_window)
    
    # Stack into tensors
    X_new = torch.stack(X_new)  # Shape: (n_windows, look_back, c, b)
    y_new = torch.stack(y_new)  # Shape: (n_windows, horizon, c, b)
    
    # Rearrange dimensions to match desired output shapes
    X_new = X_new.permute(0, 2, 1, 3)  # Shape: (n_windows, c, look_back, b)
    y_new = y_new.permute(0, 2, 1, 3)  # Shape: (n_windows, c, horizon, b)
    
    return X_new, y_new.squeeze().squeeze()

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