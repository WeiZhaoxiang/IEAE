import torch
import torch.nn as nn
from Umodel.MLP import init_he_weights
from Umodel.Unet import ActUNet 
from Umodel.EUmodel import IECE
from torch import cat, optim, abs, matmul
from PlotMap import plot_map
import yaml
from loss.lsm import LSE


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 读取cfg
    with open('/nfs/Wei/NC5/Heat_cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    device = cfg["device"]

    graph = torch.load("/nfs/Wei/NC5/data/Heat/graph.pth")
    coor = graph["nodes"].unsqueeze(0)
    x_draw = coor[0, :, 0].numpy()
    y_draw = coor[0, :, 1].numpy()

    # 读取邻接表
    adj_mat = graph["adj_mat"].to(device)

    # 读取基函数
    T_basis = graph["T_basis"].to(device) #[n, n_basis, k+1]
    T_xx = graph["T_xx"][:,:,0].unsqueeze(0).to(device) #[1, n, n_basis]
    T_yy = graph["T_yy"][:,:,0].unsqueeze(0).to(device) #[1, n, n_basis]
    # 计算基函数伪逆
    T_basis_inv = torch.linalg.pinv(T_basis)
    B_T = T_basis.permute(0,2,1)
    B = T_basis
    xs = matmul(B_T, torch.linalg.inv(matmul(B, B_T)))
    
    # mask [1, n, 1]
    mask_BC = graph["BC"].unsqueeze(0).unsqueeze(2).float().to(device)
    mask_iner = (1 - graph["BC"]).unsqueeze(0).unsqueeze(2).float().to(device)

    # 读取数据
    data_dict = torch.load("/nfs/Wei/NC5/data/Heat/pth/52.pth")
    t_arr = data_dict["temp_arr"]
    T_0 = t_arr[:, 0:1].unsqueeze(0) #[batch, n, 1]
    T_1 = t_arr[:, 1:2].unsqueeze(0) #[batch, n, 1]
    para = data_dict["para"].unsqueeze(0).to(device)

    x = cat([coor, T_0], dim=2).to(device)
    coor = coor.to(device)
    y0 = T_0.clone().to(device) #[batch, n, 1]
    y1 = T_1.clone().to(device) #[batch, n, 1]
    
    epochs = cfg["epochs"]
    e_t = 10
    lr = cfg["lr"]
    dt = cfg["dt"]
    alpha = para[0,0,0]

    # model = MLP(d_in=3, d_out=1, d=196, repeat=32)
    # model = ActUNet(
    #     d_in=3, d_out=1, d=128, 
    #     n_list=[5000, 1250, 640, 320],
    #     n_layers=12
    # )
    model = IECE(cfg)
    total_params = count_parameters(model) / 1e6
    print(f"{total_params:.2f} M")
    init_he_weights(model)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs//e_t, eta_min=2e-4
    )
    LossF = nn.SmoothL1Loss()
    l_d, l_all, l_p = 0, 0, 0
    for epoch in range(epochs):
        model.train()
        lr_now = opt.param_groups[0]['lr']

        opt.zero_grad()
        f = model(para, coor, y0)[0]
        # f = model(x)[0]
        # 数据驱动损失
        loss_d = LossF(y1*mask_BC, f*mask_BC) * f.shape[1] / mask_BC.sum()
        
        # 求解泰勒系数
        # f = y1
        A = LSE(xs, adj_mat, f) # [batch, n, n_basis]

        # 微分方程损失
        f_xx = (A*T_xx).sum(dim=2).unsqueeze(2) # [batch, n, 1]
        f_yy = (A*T_yy).sum(dim=2).unsqueeze(2) # [batch, n, 1]
        T_t = (f - y0) / dt # [batch, n, 1]
        loss_pde = LossF(T_t*mask_iner, alpha*(f_xx+f_yy)*mask_iner)
        # loss_pde = LossF(T_t, alpha*(f_xx+f_yy))

        loss_all =  loss_d + 0.001 * loss_pde 
        loss_all.backward()
        opt.step()

        l_d += loss_d.cpu().item()
        l_p += loss_pde.cpu().item()
        l_all += loss_all.cpu().item()

        if (epoch+1) % e_t == 0:
            sch.step()
            model.eval()
            with torch.no_grad():
                # f = model(para, coor, y0)[0]
                # f = model(x)[0]
                f = model(para, coor, y0)[0]
                mae = abs(f-y1).mean()
                mae = mae.cpu().item()
            
            logs = f"E:{epoch+1}/{epochs}| "
            logs += f"lr:{lr_now:.4e}| "
            logs += f"l_d:{l_d:.4e} l_p:{l_p:.4e} l_all:{l_all:.4e}"
            logs += f"| mae:{mae:.4e}"
            l_d, l_all, l_p = 0, 0, 0
            print(logs)

            if (epoch+1) % (e_t*5) == 0:
                plot_map(
                    x=x_draw, y=y_draw, z=f[0,:,0].cpu().numpy(),
                    savepath=f"./Heat2D/figures/T_{epoch+1}.png"
                )


