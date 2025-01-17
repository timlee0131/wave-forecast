import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import importlib
from termcolor import colored, cprint
import time

from models.models import SimpleLinear, MLP, CNN, TimeThenSpace
from experiments.loader import load_dataset, load_dataset_ndbc, load_dataset_graph, load_dataset_graph_cnn
from experiments.aux import loss_analysis, station_wave_information, station_wave_information_ndbc
from experiments.utils import RMSE, create_sequences, normalize_data, unnormalize_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def train_stgnn(config, model, data, verbose=False):
    train_batch, test_batch = data
    
    epochs = config.epochs
    runs = config.runs
    
    loss_list = []
    results_list = []
    
    criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss(reduction='none')
    
    min_y_pred = None   # for plotting purposes
    min_loss = float('inf')

    for run in range(runs):
        model.reset_parameters()        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()
            
            y_hat = model(train_batch.x, train_batch.edge_index)
            
            y_hat = y_hat.reshape(-1, config.num_nodes)
            target = train_batch.y.reshape(-1, config.num_nodes)
            
            loss = criterion(y_hat, target)
            
            loss.backward()
            optimizer.step()
            
            if config.epoch_verbose and epoch % (epochs // 10) == 0:
                epoch_c = colored(epoch, 'cyan')
                loss_c = colored(f'{loss.item():.2f}', 'red')
                print(f'Epoch: {epoch_c}, Loss: {loss_c}')

        model.eval()
        with torch.no_grad():
            y_hat = model(test_batch.x, test_batch.edge_index)
            
            y_hat = y_hat.reshape(-1, config.num_nodes)
            target = test_batch.y.reshape(-1, config.num_nodes)
            
            loss = eval_criterion(y_hat, target).mean(dim=0)
            
            if loss.mean() < min_loss:
                min_loss = loss.mean()
                min_y_pred = y_hat
            
            loss_list.append(loss)
            results_list.append(y_hat)

    loss_tensor = torch.stack(loss_list)
    avg_loss = loss_tensor.mean(dim=0)
    std_loss = loss_tensor.std(dim=0)
            
    if verbose:
        cprint(f'Evaluation Loss: {avg_loss}', 'yellow')
        cprint(f'standard deviation: {std_loss}', 'yellow')
    
    results_tensor = torch.stack(results_list)
    avg_results = results_tensor.mean(dim=0)
    
    return min_y_pred, test_batch.y.reshape(-1, config.num_nodes)

def driver(config_name, aux=False):
    config_path = f'./experiments/configs/group/{config_name}.py'
    config = get_config(config_path)
    
    train_batch, test_batch = load_dataset_graph_cnn(config, device)
    
    # TimeThenSpaceModel = TimeThenSpace(config.num_features * config.look_back, config.time_hidden, config.time_out, config.space_hidden, config.space_out).to(device)
    
    TimeThenSpaceModel = TimeThenSpace(config.num_features, config.time_hidden, config.time_out, config.space_hidden, config.space_out).to(device)
    
    pred, target = train_stgnn(config, TimeThenSpaceModel, [train_batch, test_batch], verbose=config.verbose)
    # print(pred)
    
    # # save the prediction
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    np.save(f'./experiments/data/npy/labn/{config_name}_pred_cnn_grouped.npy', pred_np)
    np.save(f'./experiments/data/npy/labn/{config_name}_target_cnn_grouped.npy', target_np)