import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from moviepy.editor import ImageSequenceClip
import os
import shutil

# 1. 生成示例数据
T, N = 50, 100
A = [np.column_stack([np.random.rand(N)*10, np.random.rand(N)*10, np.random.rand(N)]) for _ in range(T)]

# 2. 插值到统一网格
xi = np.linspace(0, 10, 100)
yi = np.linspace(0, 10, 100)
XI, YI = np.meshgrid(xi, yi)
vmin, vmax = 0, 1

def interpolate_frame(data):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    zi = griddata((x, y), z, (XI, YI), method='cubic')
    return zi

Z_interp = [interpolate_frame(At) for At in A]

# 3. 绘制每一帧
def plot_frame(zi, frame_idx):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(XI, YI, zi, shading='auto', norm=Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=ax, label='Z Value')
    ax.set_title(f'Frame {frame_idx}')
    plt.close(fig)
    return fig

frames = [plot_frame(Z_interp[i], i) for i in range(T)]

# 4. 保存图像并合成视频
temp_dir = "temp_frames"
os.makedirs(temp_dir, exist_ok=True)
for i, fig in enumerate(frames):
    fig.savefig(f"{temp_dir}/frame_{i:03d}.png", dpi=300)

clip = ImageSequenceClip([f"{temp_dir}/frame_{i:03d}.png" for i in range(T)], fps=10)
clip.write_videofile("heatmap_video.mp4", codec="libx264")

# 清理临时文件
shutil.rmtree(temp_dir)
