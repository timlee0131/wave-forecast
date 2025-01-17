import torch
from torch_geometric.data import Data, Batch

from experiments.utils import create_sequences_geometric

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

def load_dataset_graph(config, device):
    data = torch.load(f'{config.data_dir}/{config.dataset}.pt', weights_only=True)
    
    X_train = data['X_train'].to(device)
    X_test = data['X_test'].to(device)
    y_train = data['y_train'].to(device)
    y_test = data['y_test'].to(device)
    
    X_train, y_train = create_sequences_geometric(config, X_train, y_train, config.look_back, config.horizon)
    X_test, y_test = create_sequences_geometric(config, X_test, y_test, config.look_back, config.horizon)
    
    # flatten appropriately
    X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1], -1).to(device)
    y_train_flat = y_train.reshape(y_train.shape[0], y_train.shape[1]).to(device)

    X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1], -1).to(device)
    y_test_flat = y_test.reshape(y_test.shape[0], y_test.shape[1]).to(device)
    
    # creating graphs
    num_graphs = X_train.shape[0]
    num_nodes = X_train.shape[1]
    num_features = X_train_flat.shape[2]
    
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    train_graphs = []
    for i in range(num_graphs):
        node_features = X_train_flat[i]
        data = Data(x=node_features, edge_index=edge_index, y=y_train_flat[i])
        train_graphs.append(data)

    test_graphs = []
    for i in range(X_test.shape[0]):
        node_features = X_test_flat[i]
        data = Data(x=node_features, edge_index=edge_index, y=y_test_flat[i])
        test_graphs.append(data)
        
    train_batch = Batch.from_data_list(train_graphs).to(device)
    test_batch = Batch.from_data_list(test_graphs).to(device)
    
    print(train_batch.x.shape)
    
    return train_batch, test_batch

def load_dataset_graph_cnn(config, device):
    data = torch.load(f'{config.data_dir}/{config.dataset}.pt', weights_only=True)
    
    X_train = data['X_train'].to(device)
    X_test = data['X_test'].to(device)
    y_train = data['y_train'].to(device)
    y_test = data['y_test'].to(device)
    
    X_train, y_train = create_sequences_geometric(config, X_train, y_train, config.look_back, config.horizon)
    X_test, y_test = create_sequences_geometric(config, X_test, y_test, config.look_back, config.horizon)
    
    X_train_cnn = X_train.permute(0,1,3,2)
    X_test_cnn = X_test.permute(0,1,3,2)
    
    # X_train_cnn_grouped = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1] * X_train_cnn.shape[2], X_train_cnn.shape[3])
    # X_test_cnn_grouped = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1] * X_test_cnn.shape[2], X_test_cnn.shape[3])
    
    # creating graphs
    num_graphs = X_train.shape[0]
    num_nodes = X_train.shape[1]
    num_features = X_train.shape[3]
    
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    train_graphs = []
    for i in range(num_graphs):
        node_features = X_train_cnn[i]
        data = Data(x=node_features, edge_index=edge_index, y=y_train[i])
        train_graphs.append(data)

    test_graphs = []
    for i in range(X_test.shape[0]):
        node_features = X_test_cnn[i]
        data = Data(x=node_features, edge_index=edge_index, y=y_test[i])
        test_graphs.append(data)
    
    train_batch = Batch.from_data_list(train_graphs).to(device)
    test_batch = Batch.from_data_list(test_graphs).to(device)
    
    return train_batch, test_batch