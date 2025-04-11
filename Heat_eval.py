import torch
from torch import load, cat
import yaml
from Umodel.EUmodel import IECE
from dataloader.HeatData import HeatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from os.path import join
from PlotMap import create_vedio


def HeatEval(model:IECE, nt,  cfg=None):
    
    device = cfg["device"]

    data_dict = load(
        join(cfg["dataset"], "pth/52.pth"),
        map_location='cpu'
    )
    para = data_dict["para"].unsqueeze(0).to(device) #[1, n, d_para]
    temp_arr = data_dict["temp_arr"].unsqueeze(0).to(device) #[1, n, 51]
    T0 = temp_arr[:, :, 0:1]

    # # 加载图
    graph = torch.load(join(cfg["dataset"], "graph.pth"))
    coor = graph["nodes"].unsqueeze(0).to(device) # [1,n,2]

    # 开始推理
    res_list = []
    gt_list = []
    res_list.append(T0)
    model.eval().to(device)
    for i in range(nt):
        with torch.no_grad():
            f_list = model.forward(para, coor, res_list[-1])
            res_list.append(f_list[0])
            gt_list.append(temp_arr[:,:,i:i+1])

    A_mat = []
    for f in res_list:
        A = cat([coor, f], dim=2).cpu() # [1, n,3]
        A_mat.append(A)
    A_mat = cat(A_mat, dim=0).cpu().numpy() # [nt, n, 3]
    create_vedio(A_mat, size=[1000, 1000], vedio_path="pred.mp4")

    GT_mat = []
    for gt in gt_list:
        GT = cat([coor, gt], dim=2).cpu() # [1, n,3]
        GT_mat.append(GT)
    GT_mat = cat(GT_mat, dim=0).cpu().numpy() # [nt, n, 3]
    create_vedio(GT_mat, size=[1000, 1000], vedio_path="gt.mp4")

if __name__ == "__main__":
    model_path = "/nfs/Wei/NC5/Heat2D/checkpoints/40.pth"
    # 读取cfg
    with open('/nfs/Wei/NC5/Heat_cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    
    # 加载模型
    model = IECE(cfg)
    model.load_state_dict(load(model_path))
    

    HeatEval(model=model, nt=50, cfg=cfg)
    
