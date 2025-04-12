from Umodel.Trans import Trans, TransRes, MMNormal
from Umodel.Unet import ActUNet, Head
from Umodel.MLP import MLP
import torch.nn as nn
from torch import cat


class IECE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 读取参数
        d_para = cfg["d_para"] # 计算参数维度
        d_coor = cfg["d_coor"] # 坐标维度
        d_f = cfg["d_f"] # 解函数维度
        d_v = cfg["d_v"] # 嵌入速度维度
        n_list = cfg["n_list"] # 采样点数量
        n_act = cfg["n_act"] # 隐空间作用层数
        n_head = cfg["n_head"]

        # 计算参数嵌入
        self.para_em = nn.Sequential(
            Trans(d_para, d_v, act=nn.LeakyReLU()),
            TransRes(d_v, act=nn.Tanh()),
            Trans(d_v, d_out=n_list[-1])
        )

        # 编码Unet
        self.unet = ActUNet(
            d_in=d_coor+d_f,
            n_list=n_list,
            d=d_v, 
            n_layers=n_act
        )


        # 解码头
        self.head_list = nn.ModuleList()
        for _ in range(d_f):
            self.head_list.append(
                MLP(d_in=d_v, d_out=1, d=d_v, repeat=n_head)
            )
    
    def forward(self,
        para, # [1, 1, d_para]
        coor, # [1, n1, d_coor]
        f0, # [batch, n1, d_f]
        ):

        # 计算参数嵌入 [batch, nz, 1]
        para_em = self.para_em.forward(para).permute(0,2,1)

        # 初值合并
        batch = f0.shape[0]
        n1 = coor.shape[1]
        d_coor = coor.shape[2]
        x1 = cat([coor.expand(batch, n1, d_coor), f0], dim=2)

        # 融合
        fea = self.unet.forward(x1, para_em)

        # # 解码
        # y = self.res.forward(fea)

        # 输出
        f_list = []
        for head in self.head_list:
            f_list.append(
                head.forward(fea)
            )

        return f_list

def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:  # 只初始化权重矩阵/张量，忽略偏置和1D参数
                nn.init.xavier_normal_(param)  # 使用正态分布的Xavier初始化
                # 或者使用均匀分布的Xavier初始化:
                # nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)  # 偏置初始化为0