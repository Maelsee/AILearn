import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 显式导入3D模块

# 原始数据和函数
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) **2

# -------------------------- 关键修复：生成二维网格 --------------------------
# 1. 定义w和b的取值范围（和你原代码一致）
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

# 2. 生成二维网格矩阵（核心：把一维范围转成二维矩阵）
W, B = np.meshgrid(w_range, b_range)

# 3. 初始化loss矩阵（和W/B形状完全一致）
loss_matrix = np.zeros_like(W)

# 4. 计算每个(w,b)对应的loss（填充loss_matrix）
for i in range(len(w_range)):
    for j in range(len(b_range)):
        w = W[j, i]  # 注意meshgrid索引顺序：先b后w
        b = B[j, i]
        l_sum = 0
        for x, y in zip(x_data, y_data):
            l_sum += loss(x, y, w, b)
        loss_matrix[j, i] = l_sum / 3  # 平均损失
        # 可选：保留你原有的打印逻辑
        # print(f"w={w:.2f},b={b:.2f},loss={loss_matrix[j, i]:.4f}")

# -------------------------- 绘制3D曲面 --------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 显式指定111子图布局

# 绘制曲面（现在输入的是二维矩阵，不再报错）
surf = ax.plot_surface(W, B, loss_matrix, 
                       cmap='coolwarm',
                       alpha=0.8,
                       linewidth=0.1)

# 设置标签和标题
ax.set_xlabel("w", fontsize=12)
ax.set_ylabel("b", fontsize=12)
ax.set_zlabel("loss", fontsize=12)
ax.set_title("3D Loss Surface (Linear Model)", fontsize=14)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10, label='Loss Value')

# 强制显示图形
plt.show(block=True)
# plt.savefig("/home/meal/AI-agent/loss_surface_3d.png", dpi=150, bbox_inches='tight')