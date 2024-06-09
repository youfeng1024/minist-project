import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels : int, output_size : int):
        super(CNN, self).__init__()
        self.cnnBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, 
                    kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnnBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, 
                    kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.LazyLinear(out_features=128)
        self.fc2 = nn.LazyLinear(out_features=output_size)

        def forward(self, x):
            x = self.cnnBlock1(x)
            x = self.cnnBlock2(x)
            x = x.reshape(x.shape[0], -1)
            x = F.relu(self.fc1(x)) 
            x = self.fc2(x)
            return x