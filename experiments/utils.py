import numpy as np
import torch
import heapq

def loss_analysis(config, loss_list, k=5):
    k_largest = heapq.nlargest(k, enumerate(loss_list), key=lambda x: x[1])
    k_smallest = heapq.nsmallest(k, enumerate(loss_list), key=lambda x: x[1])
    
    indices_largest, values_largest = zip(*k_largest)
    indices_smallest, values_smallest = zip(*k_smallest)
    
    values_largest = [round(value, 4) for value in values_largest]
    values_smallest = [round(value, 4) for value in values_smallest]
    
    return indices_largest, values_largest, indices_smallest, values_smallest

def station_wave_information(config, X, verbose=False):
    station_wave_dict = {}
    
    for station in data:
        station_tensor = station['y_test']
    
        max_wave = torch.max(station_tensor)
        min_wave = torch.min(station_tensor)
        avg_wave = torch.mean(station_tensor)
        var_wave = torch.var(station_tensor)
    
        if verbose:
            print()
            print(f'Station: {station["name"]}')
            print(f'Max Wave Height: {max_wave.item():.2f} meters')
            print(f'Min Wave Height: {min_wave.item():.2f} meters')
            print(f'Average Wave Height: {avg_wave.item():.2f} meters')
            print(f'Variance Wave Height: {var_wave.item():.2f} meters')
            print()
        
        station_wave_dict[station['name']] = {
            'max_wave': max_wave.item(),
            'min_wave': min_wave.item(),
            'avg_wave': avg_wave.item(),
            'var_wave': var_wave.item()
        }
    
    return station_wave_dict
    
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

def create_sequences(config, X, y, input_steps, target_step):
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

def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))