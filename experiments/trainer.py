import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

import importlib
from termcolor import colored, cprint
import time

from models.models import SimpleLinear, MLP, CNN
from experiments.loader import load_dataset, load_dataset_ndbc
from experiments.aux import loss_analysis, station_wave_information, station_wave_information_ndbc
from experiments.utils import RMSE, create_sequences, normalize_data, unnormalize_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def train_ndbc_direct(config, model, data, is_cnn=False, verbose=False):
    runs = config.runs
    num_epochs = config.epochs
    learning_rate = config.lr
    
    total_avg_loss = 0
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    X_train, y_train = create_sequences(config, X_train, y_train, config.seq_len, config.look_ahead)
    
    X_test, y_test = create_sequences(config, X_test, y_test, config.seq_len, config.look_ahead)
    
    if is_cnn:
        X_train = X_train.transpose(1, 2)
        X_test = X_test.transpose(1, 2)
    else:
        X_train = X_train.view(X_train.shape[0], -1)
        X_test = X_test.view(X_test.shape[0], -1)
        
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    train_criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss()
    # eval_criterion = RMSE
    
    # timing 
    start_time = time.time()
    
    min_y_pred = None   # for plotting purposes
    min_loss = float('inf')
    
    loss_list = []
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        
        model.train()
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            
            loss = train_criterion(outputs.squeeze(), y_train)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            # scheduler.step()
            
            # if verbose and (epoch % (num_epochs // 10) == 0):
            if config.epoch_verbose:
                print(f'Epoch {epoch + 1}: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            
            loss = eval_criterion(y_pred.squeeze(), y_test)
            loss_list.append(loss.item())
    
            if loss.item() <= min_loss:
                min_loss = loss
                min_y_pred = y_pred
    
    end_time = time.time()
    if config.time_verbose:
        cprint(f'Time taken: {end_time - start_time:.2f} seconds', 'magenta')
    
    total_avg_loss = np.mean(loss_list)
    std_loss = np.std(loss_list)
    
    indices_largest, values_largest, indices_smallest, values_smallest = loss_analysis(config, loss_list)
    
    if verbose:
        avg_loss = colored(f'{total_avg_loss:.4f}', 'green')
        std_loss = colored(f'{std_loss:.4f}', 'light_green')
    
        print(f'Loss for station 51002: {avg_loss} +- {std_loss} meters')
        # values_lg_c = colored(values_largest, 'red')
        # values_sm_c = colored(values_smallest, 'blue')
        # print(f'10 highest loss values: {values_lg_c}')
        # print(f'10 lowest loss values: {values_sm_c}')  
    
    return min_y_pred, y_test

def train_ndbc_iter(config, model, data, is_cnn=False, verbose=False):
    runs = config.runs
    num_epochs = config.epochs
    learning_rate = config.lr
    
    total_avg_loss = 0
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    X_train, y_train = create_sequences(config, X_train, y_train, config.seq_len, config.look_ahead)
    
    X_test, y_test = create_sequences(config, X_test, y_test, config.seq_len, config.look_ahead)
    
    if is_cnn:
        X_train = X_train.transpose(1, 2)
        X_test = X_test.transpose(1, 2)
    else:
        X_train = X_train.view(X_train.shape[0], -1)
        X_test = X_test.view(X_test.shape[0], -1)
        
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    train_criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss()
    # eval_criterion = RMSE
    
    # timing 
    start_time = time.time()
    
    min_y_pred = None   # for plotting purposes
    min_loss = float('inf')
    
    loss_list = []
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        
        model.train()
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            
            loss = train_criterion(outputs.squeeze(), y_train)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            # scheduler.step()
            
            # if verbose and (epoch % (num_epochs // 10) == 0):
            if config.epoch_verbose:
                print(f'Epoch {epoch + 1}: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            i = 0
            j = config.seq_len
            X_test = X_test[i:j]
            
            for step in range(config.n_step[0]):
                y_pred = model(X_test)
                
                loss = eval_criterion(y_pred.squeeze(), y_test)
                loss_list.append(loss.item())
        
                if loss.item() <= min_loss:
                    min_loss = loss
                    min_y_pred = y_pred
                
                i += config.seq_len
                j += config.seq_len
                if j >= len(X_test):
                    break
    
    end_time = time.time()
    if config.time_verbose:
        cprint(f'Time taken: {end_time - start_time:.2f} seconds', 'magenta')
    
    total_avg_loss = np.mean(loss_list)
    std_loss = np.std(loss_list)
    
    indices_largest, values_largest, indices_smallest, values_smallest = loss_analysis(config, loss_list)
    
    if verbose:
        avg_loss = colored(f'{total_avg_loss:.4f}', 'green')
        std_loss = colored(f'{std_loss:.4f}', 'light_green')
    
        print(f'Loss for station 51002: {avg_loss} +- {std_loss} meters')
        # values_lg_c = colored(values_largest, 'red')
        # values_sm_c = colored(values_smallest, 'blue')
        # print(f'10 highest loss values: {values_lg_c}')
        # print(f'10 lowest loss values: {values_sm_c}')  
    
    return min_y_pred, y_test

def train(config, model, data, is_cnn=False, verbose=False):
    n = config.n_step[0]
    runs = config.runs
    num_epochs = config.epochs
    learning_rate = config.lr
    
    if verbose:
        if config.dataset == 'waves':
            print(f'Number of steps: {n // 2} hours')
        else:
            print(f'Number of steps: {n} hours')
        print(f'Number of runs: {runs}')
    
    for station in data:
        total_avg_loss = 0

        X_train = station['X_train'][:n * -1]
        X_test = station['X_test'][:n * -1]
        
        y_train = station['y_train'][n:]
        y_test = station['y_test'][n:]
        
        if is_cnn:
            X_train = X_train.unsqueeze(0).transpose(1, 2)
            X_test = X_test.unsqueeze(0).transpose(1, 2)
            
            # y_train = y_train.unsqueeze(0)
            # y_test = y_test.unsqueeze(0)
            
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        
        loss_list = []
        for run in range(runs):

            train_criterion = nn.MSELoss()
            # eval_criterion = nn.L1Loss()
            eval_criterion = RMSE
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            model.train()
            for epoch in range(num_epochs):
                model.train()
                outputs = model(X_train)
                loss = train_criterion(outputs.squeeze(), y_train)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                loss = eval_criterion(y_pred, y_test)
                # print(f'Final Loss: {loss.item():.4f} meters')
                loss_list.append(loss.item())
        
        total_avg_loss = np.mean(loss_list)
        std_loss = np.std(loss_list)
        
        indices_largest, values_largest, indices_smallest, values_smallest = loss_analysis(config, loss_list)
        
        if verbose:
            station_name = colored(station['name'], 'light_yellow')
            avg_loss = colored(f'{total_avg_loss:.4f}', 'green')
            std_loss = colored(f'{std_loss:.4f}', 'light_green')
        
            print(f'Loss for station {station_name}: {avg_loss} +- {std_loss} meters')
            values_lg_c = colored(values_largest, 'red')
            values_sm_c = colored(values_smallest, 'blue')
            print(f'10 highest loss values: {values_lg_c}')
            print(f'10 lowest loss values: {values_sm_c}')

def driver(config_name, aux=False):
    ndbc_whitelist = ['waves-51002', 'waves-51002-2017']
    
    config_path = f'./experiments/configs/{config_name}.py'
    
    config = get_config(config_path)
    if config.dataset in ndbc_whitelist:
        data = load_dataset_ndbc(config, device)
    else:
        data = load_dataset(config, device)
    
    if aux:
        if config.dataset in ndbc_whitelist:
            station_wave_information_ndbc(config, data, verbose=True)
        else:
            station_wave_information(config, data, verbose=True)
        return
    
    # defining models
    lin = SimpleLinear(config.num_features * config.seq_len).to(device)
    mlp = MLP(config.num_features * config.seq_len, config.mlp_hidden1, config.mlp_hidden2).to(device)
    model = CNN(config.num_features, config.cnn_hidden1, config.cnn_hidden2, config.fc_hidden, config.output_channels, config.kernel_size, config.stride).to(device)
    
    # testing iterative model
    _,_ = train_ndbc_iter(config, model, data, config.is_cnn, verbose=config.verbose)
    
    # print()
    # print(colored("training Linear...", 'blue'))
    # if config.dataset in ndbc_whitelist:
    #     y_pred_lin, y_test_lin = train_ndbc_direct(config, lin, data, verbose=config.verbose)
    # else:
    #     train(config, lin, data, verbose=config.verbose)
    
    # np.save('./experiments/data/npy/y_pred_lin.npy', y_pred_lin.cpu().numpy())
    # np.save('./experiments/data/npy/y_test_lin.npy', y_test_lin.cpu().numpy())
    
    # print()
    # print(colored("training MLP...", 'blue'))
    # if config.dataset in ndbc_whitelist:
    #     y_pred_mlp, y_test_mlp = train_ndbc_direct(config, mlp, data, verbose=config.verbose)
    # else:
    #     train(config, mlp, data, verbose=config.verbose)
    
    # np.save('./experiments/data/npy/y_pred_mlp.npy', y_pred_mlp.cpu().numpy())
    # np.save('./experiments/data/npy/y_test_mlp.npy', y_test_mlp.cpu().numpy())
    
    # print()
    # print(colored("training CNN...", 'blue'))
    # if config.dataset in ndbc_whitelist:
    #     y_pred_cnn, y_test_cnn = train_ndbc_direct(config, model, data, config.is_cnn, verbose=config.verbose)
    # else:
    #     train(config, model, data, config.is_cnn, verbose=config.verbose)
    
    # np.save('./experiments/data/npy/y_pred_cnn.npy', y_pred_cnn.cpu().numpy())
    # np.save('./experiments/data/npy/y_test_cnn.npy', y_test_cnn.cpu().numpy())