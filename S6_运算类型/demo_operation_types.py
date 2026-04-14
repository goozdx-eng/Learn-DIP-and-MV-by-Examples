"""
第六章：图像的基本运算类型 - 实验演示
==================================
实验目标：
  1. 验证点运算（灰度反转、阈值、对比度拉伸）的"只看自己"特性
  2. 对比均值滤波和中值滤波对椒盐噪声的处理效果
  3. 验证叠加原理：线性运算满足，非线性运算不满足
  4. 理解LUT（查表）加速的原理和效果

实验准备：
  pip install opencv-python numpy matplotlib

运行：
  python demo_operation_types.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 实验一：点运算——只看自己，和邻居无关
# ============================================================
print("=" * 60)
print("实验一：点运算（灰度反转 / 对比度拉伸）")
print("=" * 60)

# 创建测试图：包含渐变和色块
test_img = np.zeros((200, 400), dtype=np.uint8)
test_img[:, :200] = np.linspace(0, 255, 200, dtype=np.uint8)[np.newaxis, :]  # 左半：渐变
test_img[:, 200:] = 128  # 右半：均匀灰度

# 操作1：灰度反转（255 - 像素值）
inv_img = 255 - test_img

# 操作2：对比度拉伸（增强对比度）
# 把灰度范围从[50, 200]拉伸到[0, 255]
low, high = 50, 200
stretched = np.clip((test_img.astype(float) - low) / (high - low) * 255, 0, 255).astype(np.uint8)

# 操作3：伽马校正（提亮暗部）
gamma = 2.0
gamma_corrected = np.power(test_img.astype(float) / 255.0, 1/gamma) * 255
gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

operations = [
    (test_img, "原图"),
    (inv_img, "灰度反转\n(255-值)"),
    (stretched, "对比度拉伸\n[50,200]→[0,255]"),
    (gamma_corrected, "伽马校正\n(γ=2.0提亮暗部)"),
]

for col, (img, title) in enumerate(operations):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

    # 画灰度值曲线（第一行的中间像素的灰度值变化）
    profile = img[100, :]  # 中间行的灰度值
    axes[1, col].plot(profile, color="gray", linewidth=2)
    axes[1, col].set_ylim(0, 255)
    axes[1, col].set_title(f"第100行灰度曲线", fontsize=9)
    axes[1, col].set_xlabel("列坐标")
    axes[1, col].set_ylabel("灰度值")

plt.suptitle("实验一：点运算\n（每个输出像素只由对应的输入像素决定，和邻居无关）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S6_运算类型/实验结果_点运算.png", dpi=150, bbox_inches="tight")
print("[保存] 实验一图已保存")
plt.show()

# 【实验现象1】灰度反转后，原本暗的地方变亮，亮的地方变暗，但像素之间的相对关系保持
# 【实验现象2】对比度拉伸把原本[50,200]的范围铺开到[0,255]，暗部更暗、亮部更亮
# 【实验现象3】伽马校正非线性提亮暗部（暗部放大比例更大）

# ============================================================
# 实验二：均值滤波 vs 中值滤波对椒盐噪声的处理
# ============================================================
print("\n" + "=" * 60)
print("实验二：均值滤波 vs 中值滤波（椒盐噪声）")
print("=" * 60)

# 构造一张清晰的测试图
# 合成测试图：包含渐变和色块（不再依赖外部文件）
clear_img = np.zeros((300, 400), dtype=np.uint8)
clear_img[:, :] = np.linspace(0, 255, 400, dtype=np.uint8)[np.newaxis, :]
clear_img[100:200, 50:350] = 180
clear_img[120:180, 80:320] = 80

# 添加椒盐噪声：随机10%的像素变成0或255
noisy_img = clear_img.copy()
salt_pepper_ratio = 0.10
num_salt = int(clear_img.size * salt_pepper_ratio * 0.5)
num_pepper = int(clear_img.size * salt_pepper_ratio * 0.5)

# 随机选点加盐（白点=255）
salt_coords = np.random.randint(0, clear_img.shape[0], num_salt), \
              np.random.randint(0, clear_img.shape[1], num_salt)
noisy_img[salt_coords] = 255

# 随机选点加胡椒（黑点=0）
pepper_coords = np.random.randint(0, clear_img.shape[0], num_pepper), \
                np.random.randint(0, clear_img.shape[1], num_pepper)
noisy_img[pepper_coords] = 0

# 用不同滤波核处理
kernel_sizes = [3, 5, 7]
mean_results = {}
median_results = {}

for k in kernel_sizes:
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    mean_results[k] = cv2.filter2D(noisy_img, -1, kernel)  # 卷积（线性）
    median_results[k] = cv2.medianBlur(noisy_img, k)        # 中值（非线性）

# 计算各方法的PSNR
def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("\n[PSNR对比（越大越好，∞=完全相同）]")
print(f"{'方法':>20} | {'k=3':>8} | {'k=5':>8} | {'k=7':>8}")
print("-" * 55)
print(f"{'原图（无噪声）':>20} | {'∞':>8} | {'∞':>8} | {'∞':>8}")
for k in kernel_sizes:
    psnr_mean = psnr(clear_img, mean_results[k])
    psnr_median = psnr(clear_img, median_results[k])
    print(f"{'均值滤波 k='+str(k):>20} | {psnr_mean:>8.1f} | {psnr_mean:>8.1f} | {psnr_mean:>8.1f}")
    print(f"{'中值滤波 k='+str(k):>20} | {psnr_median:>8.1f} | {psnr_median:>8.1f} | {psnr_median:>8.1f}")

# 可视化
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

row_data = [
    ("原图（无噪声）", clear_img),
    ("加椒盐噪声", noisy_img),
    ("均值滤波 k=3", mean_results[3]),
    ("均值滤波 k=5", mean_results[5]),
    ("中值滤波 k=3", median_results[3]),
    ("中值滤波 k=5", median_results[5]),
    ("中值滤波 k=7", median_results[7]),
]

titles_and_imgs = [
    ("原图（无噪声）", clear_img),
    ("加椒盐噪声 (10%)", noisy_img),
    ("均值 k=3", mean_results[3]),
    ("均值 k=5", mean_results[5]),
    ("中值 k=3", median_results[3]),
]

for col, (title, img) in enumerate(titles_and_imgs):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

# 第二行：放大噪声区域看细节
crop_y, crop_x = 120, 80  # 文字区域的中心
crop_size = 80
for col, (title, img) in enumerate(titles_and_imgs):
    crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size*2]
    axes[1, col].imshow(crop, cmap="gray")
    axes[1, col].set_title(f"{title} (放大)", fontsize=9)
    axes[1, col].axis("off")

# 第三行：k=7对比
axes[2, 0].imshow(mean_results[7], cmap="gray")
axes[2, 0].set_title("均值 k=7 (过度模糊)", fontsize=10)
axes[2, 0].axis("off")

axes[2, 1].imshow(median_results[7], cmap="gray")
axes[2, 1].set_title("中值 k=7 (噪声去除干净)", fontsize=10)
axes[2, 1].axis("off")

# PSNR条形图
methods = ["均值k=3", "中值k=3", "均值k=5", "中值k=5", "均值k=7", "中值k=7"]
psnr_values = [
    psnr(clear_img, mean_results[3]),
    psnr(clear_img, median_results[3]),
    psnr(clear_img, mean_results[5]),
    psnr(clear_img, median_results[5]),
    psnr(clear_img, mean_results[7]),
    psnr(clear_img, median_results[7]),
]
colors = ["#ff7f7f", "#7fbf7f", "#ff7f7f", "#7fbf7f", "#ff7f7f", "#7fbf7f"]
axes[2, 2].bar(methods, psnr_values, color=colors)
axes[2, 2].set_ylabel("PSNR (dB)")
axes[2, 2].set_title("质量对比（越高越好）")
axes[2, 2].tick_params(axis='x', rotation=45)

# 留空剩余格子
axes[2, 3].axis("off")
axes[2, 4].axis("off")

plt.suptitle("实验二：均值滤波 vs 中值滤波（椒盐噪声）\n（中值滤波忽略极端值，对脉冲噪声天然有效）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S6_运算类型/实验结果_滤波对比.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# 【实验现象1】中值滤波对椒盐噪声的去除效果远好于均值滤波
# 【实验现象2】均值滤波会把噪声"糊开"（均值把极端值平均进周围像素），越大的核模糊越严重
# 【实验现象3】中值滤波的核越大，去噪越干净，但细节丢失也越多（k=7时文字边缘开始模糊）

# ============================================================
# 实验三：叠加原理验证
# ============================================================
print("\n" + "=" * 60)
print("实验三：叠加原理验证（线性 vs 非线性）")
print("=" * 60)

# 叠加原理：T[f+g] = T[f] + T[g]  且  T[k·f] = k·T[f]

# 构造两个信号
f = np.zeros((50, 50), dtype=np.float32)
f[15:35, 15:35] = 200  # 中心方块

g = np.zeros((50, 50), dtype=np.float32)
g[5:15, 5:15] = 100  # 角落小方块

# 线性运算：均值滤波
def mean_filter(img, k=3):
    kernel = np.ones((k, k)) / (k * k)
    return cv2.filter2D(img, -1, kernel)

# 非线性运算：中值滤波
def median_filter(img, k=3):
    return cv2.medianBlur(img.astype(np.uint8), k).astype(np.float32)

# 验证叠加原理
f_plus_g = f + g

# 线性运算验证
T_f = mean_filter(f, k=3)
T_g = mean_filter(g, k=3)
T_f_plus_g = mean_filter(f_plus_g, k=3)
T_f_plus_T_g = T_f + T_g

# 计算误差
linear_error = np.mean(np.abs(T_f_plus_g - T_f_plus_T_g))
print(f"[线性运算（均值滤波）叠加原理验证]")
print(f"  T[f+g] 和 T[f]+T[g] 的平均绝对误差: {linear_error:.6f}")
print(f"  {'✓ 满足叠加原理' if linear_error < 1e-5 else '✗ 不满足叠加原理'}")

# 非线性运算验证
T_f_nl = median_filter(f, k=3)
T_g_nl = median_filter(g, k=3)
T_f_plus_g_nl = median_filter(f_plus_g, k=3)
T_f_plus_T_g_nl = T_f_nl + T_g_nl

nonlinear_error = np.mean(np.abs(T_f_plus_g_nl - T_f_plus_T_g_nl))
print(f"\n[非线性运算（中值滤波）叠加原理验证]")
print(f"  T[f+g] 和 T[f]+T[g] 的平均绝对误差: {nonlinear_error:.2f}")
print(f"  {'✓ 满足叠加原理' if nonlinear_error < 1e-5 else '✗ 不满足叠加原理（这是正常的）'}")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=200)
axes[0, 0].set_title("f（中心方块）")
axes[0, 0].axis("off")

axes[0, 1].imshow(g, cmap="gray", vmin=0, vmax=200)
axes[0, 1].set_title("g（角落方块）")
axes[0, 1].axis("off")

axes[0, 2].imshow(f_plus_g, cmap="gray", vmin=0, vmax=300)
axes[0, 2].set_title("f + g")
axes[0, 2].axis("off")

axes[0, 3].imshow(T_f_plus_T_g - T_f_plus_g, cmap="gray")
axes[0, 3].set_title(f"线性叠加误差: {linear_error:.6f}\n（接近0=满足叠加原理）")
axes[0, 3].axis("off")

axes[1, 0].imshow(T_f, cmap="gray")
axes[1, 0].set_title("T[f]（均值滤波）")
axes[1, 0].axis("off")

axes[1, 1].imshow(T_g, cmap="gray")
axes[1, 1].set_title("T[g]（均值滤波）")
axes[1, 1].axis("off")

axes[1, 2].imshow(T_f_plus_g_nl, cmap="gray")
axes[1, 2].set_title("T[f+g]（中值滤波）")
axes[1, 2].axis("off")

axes[1, 3].imshow(T_f_plus_T_g_nl - T_f_plus_g_nl, cmap="gray")
axes[1, 3].set_title(f"非线性叠加误差: {nonlinear_error:.1f}\n（非零=不满足叠加原理）")
axes[1, 3].axis("off")

plt.suptitle("实验三：叠加原理验证\n（均值=线性（误差≈0）| 中值=非线性（误差非零））", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S6_运算类型/实验结果_叠加原理.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# 【实验现象】均值滤波满足叠加原理（误差≈0），中值滤波不满足（误差非零）
# 这就是"线性"和"非线性"的本质区别

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. 点运算只看自己，可用LUT查表加速（像素级独立变换）")
print("2. 均值滤波：线性（满足叠加原理），但会把噪声'糊开'")
print("3. 中值滤波：非线性（不满足叠加原理），天然忽略椒盐噪声的极端值")
print("4. 椒盐噪声→中值滤波；高斯噪声→均值/高斯滤波")
print("5. 线性运算可加速（非线性不能）：大核用频域卷积")
