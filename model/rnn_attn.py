import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_ATTN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size   
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)        
        # 定义一个可学习的参数
        self.attn_weight = nn.Linear(2 * hidden_size, 1, bias=False)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        
    def attention(self, key):             
        # 点积方式打分 计算概率分布
        weight = F.softmax(self.attn_weight(key), dim=1)
        print(weight.shape)
        value = key       
        out= torch.sum(weight * value, dim=1)
        print(out.shape)
        # (batch, hidden_size) 
        return out
        
    def forward(self, X):
        """ out的形状：(batch, seq, feature) """
        X = X.permute(0, 2, 3, 1).reshape(X.shape[0], self.input_size, -1)
        out, _ = self.bigru(X)
        # 获得所有样本最后时刻的输出
        out = self.attention(out)       
        out = self.fc(out)    
        return out