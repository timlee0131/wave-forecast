import torch

def load_dataset(config, device):
    data = torch.load(f'{config.data_dir}/{config.dataset}.pt')
    
    train_test_data = []

    for key, value in data.items():
        train_size = int(len(value['features']) * config.train_ratio)
        
        X_train, X_test = value['features'][:train_size], value['features'][train_size:]
        y_train, y_test = value['target'][:train_size], value['target'][train_size:]
        
        train_test_data.append({
            'name': key,
            'X_train': X_train.to(device),
            'X_test': X_test.to(device),
            'y_train': y_train.to(device),
            'y_test': y_test.to(device)
        })
    
    return train_test_data

def load_dataset_ndbc(config, device):
    data = torch.load(f'{config.data_dir}/{config.dataset}.pt')
    
    data['X_train'] = data['X_train'].to(device)
    data['X_test'] = data['X_test'].to(device)
    data['y_train'] = data['y_train'].to(device)
    data['y_test'] = data['y_test'].to(device)
    
    return data