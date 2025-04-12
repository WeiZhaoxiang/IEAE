import torch
from torch import load
import yaml
from model.Heat2DModel import Heat2DModel
from loss.Heat_loss import Heat2DLoss
from loss.Heat_loss2 import Heat2DLoss2
from dataloader.HeatData import HeatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np
from model.MTNet import MTNet
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch import optim


device = 3

# 加载网格
graph = torch.load(
    f"/nfs/Wei/NC5/data/Heat/graph.pth"
)
# 读取图信息
n_coor = graph["nodes"].unsqueeze(0).to(device)
T_basis = graph["T_basis"].unsqueeze(0).to(device)
# 输入数据
src_data = load("/nfs/Wei/NC5/data/Heat/pth/52.pth")
para = src_data["para"].unsqueeze(0).to(device)

Temp = src_data["T0"].unsqueeze(0).to(device)

T0 = Temp[:,:,0:1]
T1 = Temp[:,:,1:2]
epochs = 2000

# 创建损失函数--------
loss_function = Heat2DLoss2()

# 读取cfg
with open('/nfs/Wei/NC5/Heat_cfg.yaml', 'r', encoding='utf-8') as file:
    cfg = yaml.safe_load(file)
model = MTNet(cfg).to(device).train()
# 创建迭代器
opt = optim.Adam(model.parameters(), lr=cfg["lr"])
cosine_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=epochs)

for i in range(epochs):
    opt.zero_grad()
    f_list = model.forward(para, n_coor, T0)
    loss_con, loss_PDE, loss_BC, temp_c = loss_function(
        T0, f_list, para, graph, cfg["dt"]
    )
    l_all = loss_PDE + loss_BC
    l_all.backward()
    opt.step()
    cosine_sch.step()

    if (i+1) % 10 == 0:
        mae = torch.abs(temp_c - T1).mean().cpu().item()
        logs = f"l_c:{loss_con.cpu().item():.4e} "
        logs += f"l_p:{loss_PDE.cpu().item():.4e} "
        logs += f"l_b:{loss_BC.cpu().item():.4e}"
        logs += f" mae:{mae:.4e} "
        logs += f"{i+1}"
        print(logs)