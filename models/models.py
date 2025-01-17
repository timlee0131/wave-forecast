import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.optim.lr_scheduler import StepLR

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        self.reset_parameters()

    def forward(self, x):
        return self.linear(x)
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
        self.reset_parameters()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, fc_hidden, output_dim, kernel_size, stride):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim1, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim2, out_channels=output_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
        # self.mean_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        
        self.fc = nn.Linear(output_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)
        
        self.reset_parameters()

    def forward(self, x):
        t_len = x.shape[2]
        x = nn.ReLU()(self.conv1(x))
        
        x = self.mean_pool(x)
        
        x = nn.ReLU()(self.conv2(x))
        x = self.mean_pool(x)
        
        x = nn.ReLU()(self.conv3(x))
        x = self.mean_pool(x)
        
        x = x.squeeze(2)
        
        # x = F.interpolate(x, size=t_len, mode='linear', align_corners=False)
        # x = x.squeeze().transpose(0, 1)
        
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        
        return x
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset(self):
        self.reset_parameters()

# define GCN model
class SpaceGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpaceGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=8, dropout=0.6)
        # self.conv2 = GATv2Conv(hidden_dim * 8, output_dim, heads=1, concat=False, dropout=0.6)
        
        # self.linear = nn.Linear(output_dim, 1)
        self.reset_parameters()
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

class TimeCNNGrouped(nn.Module):
    def __init__(self, sets, input_dim, hidden_cnn, out_cnn, fc_hidden, kernel_size, stride):
        super(TimeCNNGrouped, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim * sets, out_channels=hidden_cnn * sets, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=hidden_cnn * sets, out_channels=out_cnn * sets, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        # self.conv2 = nn.Conv1d(in_channels=hidden_dim1 * sets, out_channels=sets * hidden_dim2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        # self.conv3 = nn.Conv1d(in_channels=hidden_dim2 * sets, out_channels=sets * output_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
        # self.mean_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(out_cnn * sets, fc_hidden * sets)
        # self.fc2 = nn.Linear(fc_hidden * sets, sets)
        
        self.reset_parameters()

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.mean_pool(x)
        
        x = nn.ReLU()(self.conv2(x))
        x = self.mean_pool(x)
        
        # x = nn.ReLU()(self.conv3(x))
        # x = self.mean_pool(x)
        
        x = x.squeeze(2)
        
        x = self.fc(x)
        # x = nn.ReLU()(x)
        # x = self.fc2(x)
        
        return x
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class TimeThenSpace(nn.Module):
    def __init__(self, input_dim, time_hidden, time_out, space_hidden, output_dim):
        super(TimeThenSpace, self).__init__()
        
        # self.time_nn = nn.Sequential(
        #     nn.Linear(input_dim, time_hidden),
        #     nn.ReLU(),
        #     nn.Linear(time_hidden, time_out)
        # )
        self.time_nn = TimeCNNGrouped(sets=5, input_dim=input_dim, hidden_cnn=32, out_cnn=48, fc_hidden=24, kernel_size=3, stride=1)
        
        self.space_nn = SpaceGNN(input_dim=24, hidden_dim=8, output_dim=output_dim)
        # self.space_nn = GCNConv(time_out, output_dim)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index):
        time_x = x.reshape(x.shape[0] // 5, x.shape[1] * 5, x.shape[2])
        time_x = self.time_nn(time_x)
        
        space_x = time_x.reshape(time_x.shape[0] * 5, time_x.shape[1] // 5)
        x = self.space_nn(space_x, edge_index)
        return x
    
    def reset_parameters(self):
        # for layer in self.time_nn:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        self.time_nn.reset_parameters()
        self.space_nn.reset_parameters()