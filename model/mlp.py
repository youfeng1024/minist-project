import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()        
        # Define linear layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hideen_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)
        
    def forward(self, X):
        # Forward pass through the network
        torch.reshape(X, (X.shape[0], X.shape[1], -1))
        self.z1 = self.fc1(X)
        self.a1 = torch.relu(self.z1)
        self.z2 = self.fc2(self.a1)
        self.a2 = torch.relu(self.z2)
        self.z3 = self.fc3(self.a2)
        return self.z3
