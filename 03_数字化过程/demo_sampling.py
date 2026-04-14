"""
第三章：图像的数字化过程 - 实验演示
==================================
实验目标：
  1. 观察采样密度（像素数量）对图像细节的影响
  2. 观察量化级别（灰度级）对渐变平滑度的影响
  3. 理解摩尔纹产生的原因——欠采样的混叠效应
  4. 理解采样和量化是两个独立维度

实验准备：
  pip install opencv-python numpy matplotlib scikit-image

运行：
  python demo_sampling.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ============================================================
# 实验一：采样密度对图像的影响（空间分辨率变化）
# ============================================================
print("=" * 60)
print("实验一：采样密度（像素数量）对图像质量的影响")
print("=" * 60)

# 加载一张细节丰富的测试图（skimage内置）
original = data.astronaut()  # 宇航员图，细节多
original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
H, W = original_gray.shape
print(f"[原始图像] 尺寸: {H}×{W} = {H*W:,} 像素")

# 构造不同采样密度的图像（通过下采样）
sampling_rates = [1.0, 0.5, 0.25, 0.125, 0.0625]  # 原始、1/2、1/4、1/8、1/16
sampled_images = []
labels = []

for rate in sampling_rates:
    new_h = int(H * rate)
    new_w = int(W * rate)
    # 下采样：用插值把大图缩成小图
    small = cv2.resize(original_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 再上采样回原尺寸，方便比较
    restored = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    sampled_images.append(restored)
    labels.append(f"采样率={rate:.2%} ({new_w}×{new_h})")
    # 计算文件大小估算
    file_size = new_h * new_w
    print(f"  {labels[-1]} → 估算像素数: {file_size:,} (原始的{rate:.2%})")

# 可视化
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (img, label) in enumerate(zip(sampled_images, labels)):
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(label, fontsize=9)
    axes[i].axis("off")

plt.suptitle("实验一：采样密度对图像质量的影响\n（同一张图，不同像素数量）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("03_数字化过程/实验结果_采样密度.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存")
plt.show()

# 【实验现象1】采样率越低，图像越模糊（棋盘格效应）
# 【实验现象2】在采样率低于25%时，细节开始明显丢失（人脸轮廓模糊）

# ============================================================
# 实验二：量化级别（灰度级）对渐变的影响
# ============================================================
print("\n" + "=" * 60)
print("实验二：量化级别（灰度级）对渐变平滑度的影响")
print("=" * 60)

# 构造灰度渐变图：从0渐变到255
gradient = np.linspace(0, 255, 512, dtype=np.uint8)
gradient_img = np.tile(gradient, (100, 1))  # 100行×512列

# 模拟不同量化级别
quantization_levels = [256, 64, 16, 4, 2]
quantized_images = []

for levels in quantization_levels:
    # 量化：把0-255映射到levels个等级
    step = 256 // levels
    quantized = (gradient_img // step) * step
    # 确保最大值是255（避免截断）
    if levels < 256:
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    quantized_images.append(quantized)
    # 计算实际有效灰度级数
    actual_levels = len(np.unique(quantized))
    print(f"  量化级={levels:3d} → 实际灰度级数: {actual_levels:3d}")

# 可视化（放大渐变区域看条带）
fig, axes = plt.subplots(2, 5, figsize=(20, 6))

for i, (img, levels) in enumerate(zip(quantized_images, quantization_levels)):
    # 上排：完整渐变
    axes[0, i].imshow(img, cmap="gray", aspect="auto")
    axes[0, i].set_title(f"灰度级={levels}", fontsize=10)
    axes[0, i].axis("off")

    # 下排：放大渐变中心区域（看条带）
    center_slice = img[:, 200:312]  # 渐变中间区域放大
    axes[1, i].imshow(center_slice, cmap="gray", aspect="auto")
    axes[1, i].set_title(f"放大条带区", fontsize=9)
    axes[1, i].axis("off")

plt.suptitle("实验二：量化级别对渐变平滑度的影响\n（上排：完整渐变 下排：放大渐变中部，看条带）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("03_数字化过程/实验结果_量化级别.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# 【实验现象1】灰度级≥64时，人眼几乎看不出条带
# 【实验现象2】灰度级≤16时，条带明显可见（灰度级=4时只有4个亮度层次）
# 【实验现象3】灰度级=2时，图像只有黑白两色，完全失去渐变信息
# 这就是"量化噪声"——原本平滑的渐变被迫变成离散的台阶

# ============================================================
# 实验三：摩尔纹模拟（欠采样的混叠效应）
# ============================================================
print("\n" + "=" * 60)
print("实验三：摩尔纹（欠采样的混叠效应）")
print("=" * 60)

# 构造高频纹理图像（模拟细条纹）
stripe_freq = 40  # 每40像素一个周期
stripes = np.zeros((300, 600), dtype=np.uint8)
x = np.arange(600)
stripes = ((x % stripe_freq) < (stripe_freq // 2)).astype(np.uint8) * 255
stripes = np.tile(stripes, (300, 1))

print(f"[高频纹理图] 条纹周期: {stripe_freq} 像素")
print(f"[问题] 用低采样率拍摄这张图，会出现摩尔纹")

# 模拟不同相机像素密度（采样率）下的拍摄效果
camera_resolutions = [600, 150, 75, 37, 19]  # 对应600/150/75/37/19的降采样比
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, res in enumerate(camera_resolutions):
    # 模拟：把高分辨率图像下采样到相机的分辨率
    # 相当于用低像素相机拍摄
    small = cv2.resize(stripes, (res, 300), interpolation=cv2.INTER_AREA)
    # 再放大回显示尺寸
    captured = cv2.resize(small, (600, 300), interpolation=cv2.INTER_NEAREST)

    axes[i].imshow(captured, cmap="gray")
    ratio = 600 / res
    axes[i].set_title(f"相机分辨率: 1/{ratio:.0f}\n(每像素={ratio:.1f}原始像素)", fontsize=9)
    axes[i].axis("off")

    # 检查是否出现摩尔纹（计算图像的高频能量）
    fft = np.fft.fft2(captured)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    # 测量离中心较远的高频分量强度
    center_h, center_w = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    high_freq_energy = np.mean(magnitude[center_h-5:center_h+5, :]) / np.mean(magnitude)
    print(f"  分辨率1/{ratio:.0f} → 高频能量比: {high_freq_energy:.2f} {'⚠️ 摩尔纹明显' if high_freq_energy > 2 else '✓ 正常'}")

plt.suptitle("实验三：欠采样产生摩尔纹\n（高频条纹被错误采样后形成低频波纹）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("03_数字化过程/实验结果_摩尔纹.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# 【实验现象1】当采样率足够高时，看到的是正确的高频条纹
# 【实验现象2】当采样率不够高（欠采样）时，高频信息折叠成低频波纹——这就是摩尔纹
# 【实验现象3】摩尔纹的频率比原始条纹低，但空间上仍然是有规律的条纹
# 这是奈奎斯特采样定理的直接体现：采样频率 < 2×信号频率 → 混叠

# ============================================================
# 实验四：采样和量化是独立的
# ============================================================
print("\n" + "=" * 60)
print("实验四：采样和量化是两个独立维度（各自独立影响图像质量）")
print("=" * 60)

# 原始图
test_img = data.camera()  # skimage内置测试图

# 组合测试：高低采样 × 高低量化
configs = [
    ("高采样+高量化", 1.0, 256),
    ("高采样+低量化", 1.0, 8),
    ("低采样+高量化", 0.25, 256),
    ("低采样+低量化", 0.25, 8),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (label, sample_rate, levels) in enumerate(configs):
    for col, (img_label, img_sample_rate, img_levels) in enumerate(configs):
        # 应用采样
        h, w = test_img.shape
        new_h, new_w = int(h * img_sample_rate), int(w * img_sample_rate)
        sampled = cv2.resize(test_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 应用量化
        if img_levels < 256:
            step = 256 // img_levels
            quantized = (sampled // step) * step
            processed = np.clip(quantized, 0, 255).astype(np.uint8)
        else:
            processed = sampled

        ax = axes[row, col]
        ax.imshow(processed, cmap="gray")
        ax.set_title(f"采样{img_sample_rate:.0%} × 量化{img_levels}级", fontsize=8)
        ax.axis("off")

plt.suptitle("实验四：采样和量化独立作用\n（行列分别是原始/低采样 × 列分别是原始/低量化）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("03_数字化过程/实验结果_采样量化独立.png", dpi=150, bbox_inches="tight")
print("[保存] 实验四图已保存")
plt.show()

# 【实验现象】低采样+高量化 = 模糊但灰度平滑；高采样+低量化 = 清晰但有块效应
# 【关键结论】两者互不影响，各自决定图像质量的不同维度

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. 采样率↓ → 空间细节丢失 → 模糊/摩尔纹（欠采样）")
print("2. 量化级↓ → 灰度层次减少 → 条带/块效应")
print("3. 采样和量化独立，互不影响，各自控制不同维度的质量")
print("4. 摩尔纹=欠采样导致高频混叠成低频，是采样定理的直接体现")
