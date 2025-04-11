from Umodel.Trans import MutiTrans, MutiTransRes, MutiLinearT
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(self, d_in, d_out, d, repeat):
        super().__init__()

        self.f_in = nn.Sequential(
            MutiTrans(d_in, d//2, c=1, act=nn.LeakyReLU()),
            MutiTrans(d//2, d//4, c=2, act=nn.LeakyReLU()),
            MutiTrans(d//4, d//4, c=4, act=nn.LeakyReLU()),
        )

        res_list = []
        for _ in range(repeat):
            res_list.append(MutiTransRes(d//4, act=nn.Tanh(), c=4))

        self.f_res = nn.Sequential(*res_list)

        self.f_out = nn.Sequential(
            MutiTrans(d//4, d//4, act=None, c=4),
            MutiTrans(d//4, d_out, act=None, c=4)
        )
    
    def forward(self, x:Tensor):
        z = self.f_in.forward(x)
        z_res = self.f_res.forward(z)
        return self.f_out(z_res)

