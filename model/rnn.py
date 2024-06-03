import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size : int, output_size : int, 
                 hidden_size: int, num_layers: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义双层 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          dropout= 0.1 , bidirectional = True ,batch_first=True)
        
        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 前向传播 RNN
        out, _ = self.rnn(x, None)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层得到最终输出
        out = self.fc(out)
        
        return out