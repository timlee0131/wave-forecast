import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import importlib
from termcolor import colored

from models.models import SimpleLinear, MLP, CNN
from experiments.loader import load_dataset, load_dataset_ndbc
from experiments.utils import loss_analysis, station_wave_information, RMSE, create_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def train_ndbc_seq(config, model, data, is_cnn=False, verbose=False):
    n = config.n_step
    runs = config.runs
    num_epochs = config.epochs
    learning_rate = config.lr
    
    total_avg_loss = 0

    # X_train = data['X_train'][:n * -1]
    # X_test = data['X_test'][:n * -1]
    
    # y_train = data['y_train'][n:]
    # y_test = data['y_test'][n:]
    
    X_train, y_train = create_sequences(config, data['X_train'], data['y_train'], config.seq_len, n)
    
    X_test, y_test = create_sequences(config, data['X_test'], data['y_test'], config.seq_len, n)
    
    if is_cnn:
        X_train = X_train.transpose(1, 2)
        X_test = X_test.transpose(1, 2)
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
    else:
        X_train = X_train.view(X_train.shape[0], -1)
        X_test = X_test.view(X_test.shape[0], -1)
        
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    # print(len(train_loader))
    # quit()
    
    train_criterion = nn.MSELoss()
    # eval_criterion = nn.L1Loss()
    eval_criterion = RMSE
    
    for run in range(runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            outputs = model(X_train)
        
            loss = train_criterion(outputs.squeeze(), y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if verbose and (epoch % (num_epochs // 10) == 0):
            if verbose:
                print(f'Epoch {epoch + 1}: {loss.item():.4f}')
        
        loss_list = []
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            loss = eval_criterion(y_pred, y_test)
            loss_list.append(loss.item())
    
    total_avg_loss = np.mean(loss_list)
    std_loss = np.std(loss_list)
    
    indices_largest, values_largest, indices_smallest, values_smallest = loss_analysis(config, loss_list)
    
    if verbose:
        avg_loss = colored(f'{total_avg_loss:.4f}', 'green')
        std_loss = colored(f'{std_loss:.4f}', 'light_green')
    
        print(f'Loss for station 51002: {avg_loss} +- {std_loss} meters')
        values_lg_c = colored(values_largest, 'red')
        values_sm_c = colored(values_smallest, 'blue')
        print(f'10 highest loss values: {values_lg_c}')
        print(f'10 lowest loss values: {values_sm_c}')  

def train_ndbc(config, model, data, is_cnn=False, verbose=False):
    n = config.n_step * 6
    runs = config.runs
    num_epochs = config.epochs
    learning_rate = config.lr
    
    total_avg_loss = 0
    
    X_train = data['X_train'][:-1]
    y_train = data['y_train'][1:]
    
    X_test = data['X_test'][:-1]
    y_test = data['y_test'][1:]
    
    if is_cnn:
        X_train = X_train.unsqueeze(0).transpose(1, 2)
        X_test = X_test.unsqueeze(0).transpose(1, 2)
        
        y_train = y_train.unsqueeze(0)
        y_test = y_test.unsqueeze(0)
    
    train_criterion = nn.MSELoss()
    # eval_criterion = nn.L1Loss()
    eval_criterion = RMSE
    
    for run in range(runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        total_loss = 0
        model.train()
        for epoch in range(num_epochs):
            model.train()
            
            outputs = model(X_train)
            loss = train_criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if verbose and (epoch % (num_epochs // 10) == 0):
                print(f'Epoch {epoch + 1}: {total_loss:.4f}')
        
        loss_list = []
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
        avg_loss = colored(f'{total_avg_loss:.4f}', 'green')
        std_loss = colored(f'{std_loss:.4f}', 'light_green')
    
        print(f'Loss for station 51002: {avg_loss} +- {std_loss} meters')
        values_lg_c = colored(values_largest, 'red')
        values_sm_c = colored(values_smallest, 'blue')
        print(f'10 highest loss values: {values_lg_c}')
        print(f'10 lowest loss values: {values_sm_c}')  

def train(config, model, data, is_cnn=False, verbose=False):
    n = config.n_step
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
            
            y_train = y_train.unsqueeze(0)
            y_test = y_test.unsqueeze(0)
        
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
                loss = train_criterion(outputs, y_train)
                
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
    ndbc_whitelist = ['waves-51002']
    
    config_path = f'./experiments/configs/{config_name}.py'
    
    config = get_config(config_path)
    if config.dataset in ndbc_whitelist:
        data = load_dataset_ndbc(config, device)
    else:
        data = load_dataset(config, device)
    
    if aux:
        station_wave_information(config, data, verbose=True)
        return
    
    # defining models
    lin = SimpleLinear(config.num_features).to(device)
    mlp = MLP(config.num_features * config.seq_len, config.hidden_channels).to(device)
    cnn = CNN(config.num_features, config.cnn_hidden, config.output_channels, config.kernel_size, config.stride).to(device)
    
    # print()
    # print(colored("training Linear...", 'blue'))
    # if config.dataset in ndbc_whitelist:
    #     train_ndbc(config, lin, data, verbose=config.verbose)
    # else:
    #     train(config, lin, data, verbose=config.verbose)
    
    # print()
    # print(colored("training MLP...", 'blue'))
    # if config.dataset in ndbc_whitelist:
    #     train_ndbc_seq(config, mlp, data, verbose=config.verbose)
    # else:
    #     train(config, mlp, data, verbose=config.verbose)
    
    print()
    print(colored("training CNN...", 'blue'))
    if config.dataset in ndbc_whitelist:
        train_ndbc_seq(config, cnn, data, config.is_cnn, verbose=config.verbose)
    else:
        train(config, cnn, data, config.is_cnn, verbose=config.verbose)