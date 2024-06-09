import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, seq_len : int, seq_d : int, output_size : int, 
                 hidden_size: int, num_layers: int):
        super(RNN, self).__init__()
        self.seq_len = seq_len
        self.seq_d = seq_d
        self.hidden_size = hidden_size
        self.num_layers = num_layers     
        # 定义双层 RNN 层
        self.rnn = nn.RNN(self.seq_len, self.hidden_size, num_layers, 
                          dropout= 0.1 , bidirectional = True ,batch_first=True)    
        # 定义输出层
        self.fc = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).reshape(-1, self.seq_len, self.seq_d)
        out, _ = self.rnn(x, None)   
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        # 通过全连接层得到最终输出
        out = self.fc(out)     
        return out