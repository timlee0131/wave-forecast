import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x).squeeze()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

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
        # x = x.transpose(0,1)
        
        return x
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset(self):
        self.reset_parameters()