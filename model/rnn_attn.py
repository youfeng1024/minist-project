import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_ATTN(nn.Module):
    def __init__(self):
        super().__init__()        
        self.gru = nn.GRU(28, 64, batch_first=True)
        self.fc = nn.Linear(64, 10)        
        # 定义一个可学习的参数
        self.query = nn.Parameter(torch.zeros(64))
        
    def attention(self, gru_out):
        # gru_out: (batch, seq, hidden_size)                
        # 点积方式打分
        score = torch.matmul(gru_out, self.query)
        # (batch, seq) 
        # 计算概率分布
        A = F.softmax(score, 1)
        # (batch, seq)
        A = A.unsqueeze(1)
        # (batch, 1, seq)             
        out = torch.matmul(A, gru_out)
        # (batch, 1, hidden_size)
        out = out.squeeze(1)
        # (batch, hidden_size) 
        return out
        
    def forward(self, X):
        """ out的形状：(batch, seq, feature) """
        out, _ = self.gru(X)   
        # 获得所有样本最后时刻的输出
        out2 = out[:, -1, :]  # (batch, hidden_size)   
        # 计算各个时序的注意力分布
        # out2 = self.attention(out)        
        x = self.fc(out2)    
        return x