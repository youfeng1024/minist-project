import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer_dims: list[int]):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # 最后一层不需要激活函数
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 展平输入
        return self.network(x)