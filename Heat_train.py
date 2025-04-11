import torch
from torch import optim
import torch.nn as nn
import yaml
from loss.lsm import LSE
from dataloader.HeatData import HeatDataset, clear_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
import random
import numpy as np
from Umodel.EUmodel import IECE
import copy
from loss.HeatLoss import HeatLoss

# 设定固定种子
seed = 201302
# 1. 设置Python随机种子
random.seed(seed)
# 2. 设置Numpy随机种子
np.random.seed(seed)
# 3. 设置PyTorch随机种子
torch.manual_seed(seed)
# 4. 配置CuDNN以增加确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 5. 固定DataLoader的随机性
def seed_worker(worker_id):
    # 每个worker的种子基于初始种子和worker id生成
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# 创建生成器并设置种子
g = torch.Generator()
g.manual_seed(seed)


def train(cfg):
    # 训练设备
    device = cfg["device"]
    # 加载图------------------------------------
    graph = torch.load(join(cfg["dataset"], "graph.pth"))
    coor = graph["nodes"].unsqueeze(0).to(device)
    # 创建数据集
    dataset = HeatDataset(cfg)
    # 创建模型
    model = IECE(cfg).to(device)
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1000000}M \n")
    # 创建迭代器
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    # 创建学习率调制器--------
    cosine_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=opt, 
        T_0=cfg["T_warm"],
        T_mult=1,
        eta_min=1e-5
    )
    # 创建损失函数--------
    LossF = nn.SmoothL1Loss(reduction='none')
    heat_loss = HeatLoss(graph, cfg, loss_f=LossF, device=device)

    # 开始训练 ------------------
    clear_folder(join(cfg["savepath"], "checkpoints"))
    for epoch in range(cfg["epochs"]):
        iters = 0
        l_c = 0
        l_p = 0
        num_new = 0
        # 更新迭代器
        data_loader = DataLoader(
            copy.deepcopy(dataset), 
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=16,
            worker_init_fn=seed_worker,
            generator=g
        )
        lr_now = opt.param_groups[-1]['lr']
        model.to(device).train()
        # 开始迭代
        for para, T0, data_id in tqdm(data_loader, ncols=120):
            # 前向传播，自动混合精度
            para = para.to(device)
            T0 = T0.to(device)
            opt.zero_grad()
            f_list = model.forward(para, coor, T0)
            f = f_list[0]
            # 计算损失函数
            loss_con, loss_PDE = heat_loss.forward(
                f, T0, alpha=para[:,:,0:1]
            )
            l_all = loss_con.mean() + 0.001 * loss_PDE.mean()
            # 反向传播与优化
            l_all.backward()
            opt.step()
            # 统计损失函数
            l_c += loss_con.mean().cpu().item()
            l_p += loss_PDE.mean().cpu().item()

            # 保存合理的数值-------------------------------------
            # 计算batch中每个样本的损失
            lc_p = loss_con.mean(dim=1).squeeze(1) # [batch]
            lp_p = loss_PDE.mean(dim=1).squeeze(1) # [batch]
            # 索引出损失小于阈值的样本
            idx_ev = (lc_p<cfg["thre_B"]) & (lp_p<cfg["thre_P"])
            para_in = para[idx_ev, :, :]
            # 如果没有合理的样本，则跳过
            if para_in.size(0) < 1:
                continue
            # 更新缓冲区
            num_new += para_in.size(0)
            f_in = f[idx_ev, :, :]
            id_in = data_id[idx_ev, :]
            for ch in range(para_in.size(0)):
                para_c = para_in[ch, :, :]
                f_c = f_in[ch, :, :]
                id_c = id_in[ch, :]
                data_dict = {
                    "para": para_c.cpu().detach().float(),
                    "temp_arr": f_c.cpu().detach().float()
                }
                dataset.update_buff(data_dict, id_c)
            # 当前迭代次数
            iters += 1
        # 更新数据集列表
        dataset.update_data()
        # 学习率设置
        cosine_sch.step()
        
        # 打印log
        logs = f"E: {epoch+1}/{cfg['epochs']} |"
        logs += f"l_c:{l_c/iters:.3e}  l_p:{l_p/iters:.3e} |"
        logs += f" update:{num_new} samples:{len(dataset)} |"
        logs += f" lr:{lr_now:.3e} \n"
        print(logs)

        # 保存当前模型
        if (epoch+1) % cfg["save_interval"] == 0:
            torch.save(
                model.float().eval().cpu().state_dict(),
                join(
                    cfg["savepath"], 
                    f"checkpoints/{epoch+1}.pth"
                )
            )
        
if __name__ == "__main__":
    # 读取cfg
    with open('/nfs/Wei/NC5/Heat_cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    train(cfg)