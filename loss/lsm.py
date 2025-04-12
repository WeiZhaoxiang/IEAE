import torch
from torch import matmul, Tensor

def LSE(
    basis_inv: Tensor, #[n, k+1, n_basis], 
    adj_mat: Tensor, #[n, k+1]
    f: Tensor #[batch, n, 1]
    ):

    # 提取出自身及邻域数值
    batch = f.shape[0]
    n = f.shape[1]
    k = adj_mat.shape[1] - 1
    adj_idx = adj_mat.unsqueeze(0).expand(batch, n, k+1)
    f_near = torch.gather(
        f.expand(batch, n, k+1),
        dim=1,
        index=adj_idx
    ).unsqueeze(2) # [batch, n, 1, k+1]

    # 求解原始矩阵
    A_z = matmul(f_near, basis_inv.unsqueeze(0)) # [batch, n, 1, n_basis]
    A = A_z.squeeze(2) # [batch, n, n_basis]

    return A



