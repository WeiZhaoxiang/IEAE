import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
import os
import shutil
from moviepy.editor import ImageSequenceClip


def plot_map(
    x:np.ndarray, #[n]
    y:np.ndarray, #[n]
    z:np.ndarray, #[n]
    title = "HeatMap",
    savepath = "./test.png"
    ):

    # 创建网格
    xi = np.linspace(x.min(), x.max(), 256)
    yi = np.linspace(y.min(), y.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    # 插值（线性/三次样条等）
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi[zi<0] = 0
    # 绘制
    plt.figure(dpi=300)
    contour = plt.contourf(
        xi, yi, zi, 
        levels=40, cmap='viridis'
    )
    plt.colorbar(contour, label='Value')
    # plt.scatter(x, y, c='k', s=5, alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    # plt.show()
    plt.savefig(savepath)
    plt.close()

def interpolate_frame(data, XI, YI):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    zi = griddata((x, y), z, (XI, YI), method='cubic')
    return zi

def plot_frame(zi, frame_idx, XI, YI, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(
        XI, YI, zi, shading='auto', norm=Normalize(vmin=vmin, vmax=vmax)
    )
    fig.colorbar(im, ax=ax, label='Z Value')
    ax.set_title(f'Frame {frame_idx}')
    # # 设置坐标轴范围为 XI 和 YI 的最小最大值
    # ax.set_xlim(XI.min(), XI.max())
    # ax.set_ylim(YI.min(), YI.max())
    ax.set_aspect('equal')  # 关键设置
    plt.close(fig)
    return fig

def create_vedio(
    A_mat:np.ndarray, # [T, n, 3]
    size, # [h, w]
    vedio_path
    ):
    v_min = 0
    v_max = A_mat.max()
    nt = A_mat.shape[0]
    # 1. 插值到统一网格
    coor_x = A_mat[0,:,0]
    coor_y = A_mat[0,:,1]
    xi = np.linspace(coor_x.min(), coor_x.max(), size[0])
    yi = np.linspace(coor_y.min(), coor_y.max(), size[1])
    XI, YI = np.meshgrid(xi, yi)
    Z_interp = []
    for i in range(nt):
        At = A_mat[i, :, :]
        Z_interp.append(interpolate_frame(At, XI, YI))

    # 3. 绘制每一帧
    frames = [plot_frame(Z_interp[i], i, XI, YI, v_min, v_max) for i in range(nt)]

    # 4. 保存图像并合成视频
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    for i, fig in enumerate(frames):
        fig.savefig(f"{temp_dir}/frame_{i:03d}.png", dpi=300)

    clip = ImageSequenceClip(
        [f"{temp_dir}/frame_{i:03d}.png" for i in range(nt)], fps=5
    )
    clip.write_videofile(vedio_path, codec="libx264")

    # 清理临时文件
    shutil.rmtree(temp_dir)