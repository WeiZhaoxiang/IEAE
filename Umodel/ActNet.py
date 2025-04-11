from Umodel.Trans import Trans, TransRes, MMNormal
import torch
import torch.nn as nn
from torch import Tensor, matmul


class ActAttention(nn.Module):
    def __init__(self,
        d, # 特征维度
        ):
        super().__init__()

        self.fQ = TransRes(d, act=nn.LeakyReLU())
        self.fK = TransRes(d, act=nn.LeakyReLU())
        self.fV = TransRes(d, act=nn.LeakyReLU())

    def forward(self, 
        x:Tensor # [batch, n, d]
        ):

        # [batch, n, d]
        Q_s = self.fQ.forward(x)
        K_s = self.fK.forward(x) 
        V = self.fV.forward(x)
        # 归一化参数 [batch, 1, 1]
        r_Q = torch.norm(Q_s, p=2, dim=2).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        r_K = torch.norm(K_s, p=2, dim=2).max(dim=1)[0].unsqueeze(1).unsqueeze(1)

        Q = Q_s / (r_Q+1e-6)
        K = K_s / (r_K+1e-6)
        Q_T = Q.permute(0, 2, 1)
        K_T = K.permute(0, 2, 1)

        # 计算
        y = matmul(Q, matmul(K_T, V)) - matmul(K, matmul(Q_T, V))

        return y

class ActRes(nn.Module):
    def __init__(self,
        d, # 特征维度
        ):
        super().__init__()
        self.actAtt = ActAttention(d)
    
    def forward(self, x:Tensor):
        res = self.actAtt.forward(x)
        return MMNormal(x + res)

class ActPipe(nn.Module):
    def __init__(self,
        d, # 特征维度
        n_act # 作用次数
        ):
        super().__init__()
        act_list = []
        for _ in range(n_act):
            act_list.append(ActRes(d))
        
        self.pipe = nn.Sequential(*act_list)
    
    def forward(self, x:Tensor):
        return self.pipe.forward(x)