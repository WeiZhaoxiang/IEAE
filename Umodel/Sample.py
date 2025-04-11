from Umodel.Trans import Trans, TransRes, MMNormal
import torch
import torch.nn as nn
from torch import Tensor, cat, matmul


class SampleM(nn.Module):
    def __init__(self,
        n_out, # 下采样点的维度
        d, # 特征维度
        ):
        super().__init__()
        # K值函数
        self.fK = TransRes(d, act=nn.Tanh())
        # 采样核函数
        self.S = nn.Parameter(
            torch.randn([1, n_out, d])
        )
    
    def forward(self, 
        x:Tensor #[batch, n_in, d]
        ):
        # 计算K值
        K = self.fK.forward(x) #[batch, n_in, d]
        KT = K.permute(0,2,1) #[batch, d, n_in]
        # 计算归一化系数

        #[batch, n_out, 1]
        r_S = torch.norm(self.S, p=2, dim=2).unsqueeze(2)
        #[batch, 1, 1]
        r_Smax = r_S.max(dim=1)[0].unsqueeze(1) + 1e-5
        #[batch, n_in, 1]
        r_K = torch.norm(K, p=2, dim=2).unsqueeze(2)
        #[batch, 1, 1]
        r_Kmax = r_K.max(dim=1)[0].unsqueeze(1) + 1e-5

        return self.S / r_Smax, KT / r_Kmax

class DownSample(nn.Module):
    def __init__(self,
        n_out, # 下采样点的维度
        d, # 特征维度
        ):
        super().__init__()
        self.sample = SampleM(n_out, d)
    
    def forward(self,
        x:Tensor #[batch, n_in, d]
        ):
        # S[1, n_out, d]  KT[batch, d, n_in]
        S, KT = self.sample.forward(x)
        y = matmul(S, matmul(KT, x)) #[batch, n_out, d]
        return MMNormal(y) #[batch, n_out, d]

# class UpSample(nn.Module):
#     def __init__(self,
#         n_in, # 输入特征的维度
#         d, # 特征维度
#         ):
#         super().__init__()
#         self.sample = SampleM(n_in, d)
    
#     def forward(self,
#         x_in:Tensor, #[batch, n_in, d]
#         x_s:Tensor, #[batch, n_out, d]
#         ):
#         S, KT = self.sample.forward(x_s)
#         ST = S.permute(0, 2, 1) # [1, d, n_in]
#         K = KT.permute(0, 2, 1) # [batch, n_out, d]

#         # [batch, n_out, d]
#         y = MMNormal(matmul(K, matmul(ST, x_in)))

#         # #[batch, n_out, 2*d]
#         # zy = cat([x_s, y], dim=2)

#         return x_s + y

class UpSample(nn.Module):
    def __init__(self,
        d, # 特征维度
        ):
        super().__init__()
        self.fQ = TransRes(d, nn.Tanh())
        self.fK = TransRes(d, nn.Tanh())
    
    def forward(self,
        x_in:Tensor, #[batch, n_in, d]
        x_s:Tensor, #[batch, n_out, d]
        ):
        # 采样矩阵
        Q = self.fQ.forward(x_s) #[batch, n_out, d]
        K = self.fK.forward(x_in) #[batch, n_in, d]
        KT = K.permute(0,2,1)

        #[batch, n_out, 1]
        r_Q = torch.norm(Q, p=2, dim=2).unsqueeze(2)
        #[batch, 1, 1]
        r_Qmax = r_Q.max(dim=1)[0].unsqueeze(1) + 1e-5
        #[batch, n_in, 1]
        r_K = torch.norm(K, p=2, dim=2).unsqueeze(2)
        #[batch, 1, 1]
        r_Kmax = r_K.max(dim=1)[0].unsqueeze(1) + 1e-5

        y = matmul(Q/r_Qmax, matmul(KT/r_Kmax, x_in))

        return MMNormal(y)