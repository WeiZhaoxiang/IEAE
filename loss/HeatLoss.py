from torch import matmul, Tensor
from loss.lsm import LSE
import torch.nn as nn

class HeatLoss(nn.Module):
    def __init__(self,
        graph,
        cfg,
        loss_f,
        device
        ):
        super().__init__()
        # 步长
        self.dt = cfg["dt"]
        # 读取图
        self.B_inv = graph["T_basis_inv"].to(device)
        self.T_xx = graph["T_xx"][:,:,0].unsqueeze(0).to(device) #[1, n, n_basis]
        self.T_yy = graph["T_yy"][:,:,0].unsqueeze(0).to(device) #[1, n, n_basis]
        # 加载 mask [1, n, 1]
        self.mask_BC = graph["BC"].unsqueeze(0).unsqueeze(2).float().to(device)
        self.mask_iner = (1 - graph["BC"]).unsqueeze(0).unsqueeze(2).float().to(device)
        # 读取邻接表
        self.adj_mat = graph["adj_mat"].to(device)
        # 损失函数
        self.loss_f = loss_f

    def forward(self, f:Tensor, y0:Tensor, alpha:Tensor):
        # 边界损失
        loss_con = self.loss_f(y0*self.mask_BC, f*self.mask_BC) * \
             f.shape[1] / self.mask_BC.sum()
        
        # 求解泰勒系数
        A = LSE(self.B_inv, self.adj_mat, f) # [batch, n, n_basis]

        # 微分方程损失
        f_xx = (A*self.T_xx).sum(dim=2).unsqueeze(2) # [batch, n, 1]
        f_yy = (A*self.T_yy).sum(dim=2).unsqueeze(2) # [batch, n, 1]
        T_t = (f - y0) / self.dt # [batch, n, 1]
        loss_PDE = self.loss_f(
            T_t*self.mask_iner, alpha*(f_xx+f_yy)*self.mask_iner
        )

        return loss_con, loss_PDE