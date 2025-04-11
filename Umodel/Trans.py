import torch
import torch.nn as nn
from torch import Tensor, matmul

def MMNormal(x:Tensor)->Tensor: 
    # x: [batch, nodes, d])
    # 计算长度 [batch, nodes, 1]
    x_L2 = torch.norm(x, p=2, dim=-1).unsqueeze(-1)
    # 计算均值 [batch, 1, 1]
    L2_mean = x_L2.mean(dim=-2).unsqueeze(-2)
    return x / (L2_mean+1e-4)

class LinearT(nn.Module):
    def __init__(self,
        d_in, # 输入特征维度
        d_out, # 输入特征维度
        ):
        super().__init__()
        # 参数矩阵
        self.W = nn.Parameter(
            torch.randn([1, d_in, d_out]) # 
        )
        # 偏置向量
        self.bias = nn.Parameter(
            torch.randn([1, 1, d_out]) # 
        )
    
    def forward(self,
        x:Tensor # [batch, nodes, d_in]
        )->Tensor:
        # 投影
        y = matmul(x, self.W) + self.bias
        return y # [batch, nodes, d_out]

class Trans(nn.Module):
    def __init__(self,
        d_in, # 输入特征维度
        d_out, # 输入特征维度
        act=None # 激活函数
        ):
        super().__init__()
        # 线性变换
        self.linear = LinearT(d_in, d_out)
        # 激活函数
        self.act = act
    
    def forward(self,
        x:Tensor # [batch, nodes, d_in]
        )->Tensor:
        # 线性投影
        x_l = self.linear.forward(x) # [batch, nodes, d_out]
        # 激活
        if self.act is None:
            y = x_l
        else:
            y = self.act(MMNormal(x_l))
        return y #[batch, nodes, d_out]

class TransRes(nn.Module):
    def __init__(self,
        d, # 特征维度
        act=None # 激活函数
        ):
        super().__init__()
        # 单次变换
        self.trans = Trans(d, d, act)
    
    def forward(self, 
        x:Tensor # [batch, nodes, d]
        )->Tensor:
        # 线性变换
        res = self.trans.forward(x)
        return x + res


class MutiLinearT(nn.Module):
    def __init__(self,
        d_in, # 输入特征维度
        d_out, # 输入特征维度
        c # 分组的数量
        ):
        super().__init__()
        # 参数矩阵
        self.W = nn.Parameter(
            0.2 * torch.randn([c, 1, d_in, d_out]) # 
        )
        # 偏置向量
        self.bias = nn.Parameter(
            0.2 * torch.randn([c, 1, 1, d_out]) # 
        )
    
    def forward(self,
        x:Tensor # [*, batch, nodes, d_in]
        )->Tensor:
        # 投影
        y = matmul(x, self.W) + self.bias
        return y # [c, batch, nodes, d_out]

class MutiTrans(nn.Module):
    def __init__(self,
        d_in, # 输入特征维度
        d_out, # 输入特征维度
        c, # 分组的数量
        act=None # 激活函数
        ):
        super().__init__()
        # 线性变换
        self.linear = MutiLinearT(d_in, d_out, c)
        # 聚合函数
        self.comb = nn.Parameter(
            torch.randn([c, 1, 1, 1])
        )
        self.act = act
    
    def forward(self,
        x:Tensor # [batch, nodes, d_in]
        )->Tensor:
        # 线性投影
        x_l = self.linear.forward(
            x.unsqueeze(0)
        ) # [1, batch, nodes, d_out]
        # 激活
        if self.act is None:
            x_a = x_l
        else:
            x_a = self.act(
                MMNormal(x_l)
            )
        # 聚合
        y = (x_a*self.comb).mean(dim=0)
        return y #[batch, nodes, d_out]

class MutiTransRes(nn.Module):
    def __init__(self,
        d, # 特征维度
        c, # 分组的数量
        act=None # 激活函数
        ):
        super().__init__()
        # 单次变换
        self.trans = MutiTrans(d,d,c,act)
    
    def forward(self, 
        x:Tensor # [batch, nodes, d]
        )->Tensor:
        # 线性变换
        res = self.trans.forward(x)
        return x + res