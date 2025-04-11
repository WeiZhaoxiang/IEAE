import os
import torch
from torch.utils.data import Dataset
from os.path import join
import os
import shutil
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import copy
# 设定固定种子
seed = 201302
# 1. 设置Python随机种子
random.seed(seed)

def clear_folder(path):
    # 清空文件夹
    shutil.rmtree(path)
    os.makedirs(path)

def init_buffer(cfg):
    pth_folder = join(cfg["dataset"], "pth")
    buff_folder = join(cfg["savepath"], "buff")
    clear_folder(buff_folder)
    # 初值
    file_list = os.listdir(pth_folder)
    for i in tqdm(range(len(file_list)), ncols=80):
        pth_file = join(pth_folder, f"{i+1}.pth")
        buff_file = join(buff_folder, f"{i}_0.pth")
        data_dict = torch.load(
            pth_file, map_location="cpu"
        )
        buf_dict = {
            "para": data_dict["para"],
            "temp_arr": data_dict["temp_arr"][:, 0:1].clone()
        }
        torch.save(buf_dict, buff_file)
    return len(file_list)


class HeatDataset(Dataset):
    def __init__(self, cfg):
        # 初始化缓冲区文件
        n_para = init_buffer(cfg)
        # 初始化缓冲区表
        self.b_max = cfg["buff_max"]
        self.buff_flag = torch.zeros([n_para, self.b_max]).long()
        self.buff_flag[:, 0] = 1

        # 初始化缓冲区列表
        self.buff_list = []
        for i in range(n_para):
            data_id = torch.Tensor([i, 0]).long()
            self.buff_list.append(data_id)

        # 缓冲文件路径
        self.buff_folder = join(cfg["savepath"], "buff")

        # 每轮最大的样本量
        self.max_samples = cfg["batch"] * cfg["max_iters"]

        # 更新阈值
        self.thre_B = cfg["thre_B"]
        self.thre_P = cfg["thre_P"]

        # 完整迭代标记
        self.full_flag = True

        # 测试的batch
        self.test_batch = cfg["batch"] * 4
        self.nw = cfg["num_workers"]

    def __len__(self):
        # 限制最大迭代次数
        if self.full_flag:
            return len(self.buff_list)
        else:
            return min(self.max_samples, len(self.buff_list))
    
    def update_buff(self, 
        new_dict,  # dict
        id_c: torch.Tensor # [2]
        ):
        id_new = id_c.clone()
        id_new[1] = id_c[1] + 1
        # 超出缓冲区大小则跳过
        if id_new[1] >= self.b_max:
            return
        # 没有的更新列表
        if self.buff_flag[id_new[0], id_new[1]] < 1:
            self.buff_list.append(id_new)
        self.buff_flag[id_new[0], id_new[1]] = 1
        # 保存文件
        file_name = join(
            self.buff_folder,
            f"{id_new[0].item()}_{id_new[1].item()}.pth"
        )
        torch.save(new_dict, file_name)

    def update_data(self):
        random.shuffle(self.buff_list)

    def update_all(self, model, coor, loss, device):
        n_last = self.__len__()
        thre_B = self.thre_B
        thre_P = self.thre_P
        # 创建迭代器
        self.full_flag = True # 完整更新
        data_loader = DataLoader(
            copy.deepcopy(self), 
            batch_size=self.test_batch,
            shuffle=False,
            num_workers=self.nw,
        )
        with torch.no_grad():
            for para, T0, data_id in tqdm(data_loader, ncols=100, desc="Update:"):
                # 前向传播，自动混合精度
                para = para.to(device)
                T0 = T0.to(device)
                f_list = model.forward(para, coor, T0)
                f = f_list[0]
                # 计算损失
                loss_con, loss_PDE = loss.forward(
                    f, T0, alpha=para[:,:,0:1]
                )
                # 更新缓冲区
                # 计算batch中每个样本的损失
                lc_p = loss_con.mean(dim=1).squeeze(1) # [batch]
                lp_p = loss_PDE.mean(dim=1).squeeze(1) # [batch]
                # 索引出损失小于阈值的样本
                idx_ev = (lc_p<thre_B) & (lp_p<thre_P)
                para_in = para[idx_ev, :, :]
                # 如果没有合理的样本，则跳过
                if para_in.size(0) < 1:
                    continue
                # 更新缓冲区
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
                    self.update_buff(data_dict, id_c)
        self.update_data()
        n_now = self.__len__()
        return n_now, n_now-n_last

    def __getitem__(self, idx):
        data_id = self.buff_list[idx] # [2]
        file_name = join(
            self.buff_folder,
            f"{data_id[0].item()}_{data_id[1].item()}.pth"
        )

        data_dict = torch.load(
            file_name,
            map_location='cpu'
        )
        para = data_dict["para"]
        T0 = data_dict["temp_arr"]
        return para, T0, data_id



