from Umodel.Trans import Trans, TransRes, MMNormal
from Umodel.Sample import DownSample, UpSample
from Umodel.ActNet import ActPipe
from Umodel.MLP import MLP
import torch.nn as nn
from torch import Tensor

class Head(nn.Module):
    def __init__(self, 
        d, # 特征维度
        ):
        super().__init__()
        self.head = MLP(d_in=d, d_out=1, d=d, repeat=8)
    def forward(self, x:Tensor):
        return self.head(x)

class ActUNet(nn.Module):
    def __init__(self,
        d_in, # 输入维度
        n_list, # 下采样点数列表
        d, # 特征维度
        n_layers, # 作用层数量 
        ):
        super().__init__()
        # 采样点数量
        n2 = n_list[0]
        n3 = n_list[1]
        n4 = n_list[2]
        n5 = n_list[3]

        # 输入层
        self.l1 = Trans(d_in, d, act=None)

        # 下采样层
        self.d1 = DownSample(n_out=n2, d=d)
        self.d2 = DownSample(n_out=n3, d=d)
        self.d3 = DownSample(n_out=n4, d=d)
        self.d4 = DownSample(n_out=n5, d=d)
        self.r1 = TransRes(d, act=nn.Tanh())
        self.r2 = TransRes(d, act=nn.Tanh())
        self.r3 = TransRes(d, act=nn.Tanh())
        self.r4 = TransRes(d, act=nn.Tanh())

        # 作用层
        self.act = ActPipe(d, n_layers)

        # 上采样层
        self.up1 = UpSample(d=d)
        self.up2 = UpSample(d=d)
        self.up3 = UpSample(d=d)
        self.up4 = UpSample(d=d)


    def forward(self, 
        x1:Tensor, # [batch, n1, d_in]
        para_em:Tensor #[batch, nz, 1]
        ):

        # 下采样过程
        z1 = self.l1(x1)
        z2 = self.r1(self.d1(z1))
        z3 = self.r2(self.d2(z2))
        z4 = self.r3(self.d3(z3))
        z5 = self.r4(self.d4(z4)) # [batch, nz, d]

        # 作用
        y1 = self.act(z5*para_em) 

        # 上采样过程
        u1 = z5 + y1
        u2 = self.up1.forward(u1, z4) + z4
        u3 = self.up2.forward(u2, z3) + z3
        u4 = self.up3.forward(u3, z2) + z2
        u5 = self.up4.forward(u4, z1) + z1

        return u5


