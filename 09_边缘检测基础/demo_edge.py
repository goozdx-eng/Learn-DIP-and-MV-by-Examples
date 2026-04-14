"""
第九章：边缘检测与分割基础 - 实验演示
==================================
实验目标：
  1. 观察三种差分算子（Roberts/Sobel/Prewitt）的边缘检测效果
  2. 理解梯度幅值和方向的概念
  3. 验证Sobel对噪声的敏感性和高斯平滑的必要性
  4. 观察阈值分割的不同方法（固定/Otsu/自适应）及其适用场景
  5. 理解Otsu在单峰直方图上失效的原因

实验准备：
  pip install opencv-python numpy matplotlib

运行：
  python demo_edge.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 实验一：三种差分算子的边缘检测对比
# ============================================================
print("=" * 60)
print("实验一：Roberts / Sobel / Prewitt 边缘检测算子对比")
print("=" * 60)

# 加载测试图
test_path = "05_文件格式/test_format.png"
test_gray = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
if test_gray is None:
    # 合成测试图：包含不同方向的边缘
    test_gray = np.zeros((300, 400), dtype=np.uint8)
    # 加垂直边缘
    test_gray[:, 100:110] = 200
    test_gray[:, 200:220] = 100
    # 加水平边缘
    test_gray[80:90, :] = 200
    test_gray[180:200, :] = 50
    # 加斜线
    for i in range(300):
        j = int(i * 1.2)
        if j < 400:
            test_gray[i, j] = 180

# Roberts算子（手动实现）
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
G_x = cv2.filter2D(test_gray.astype(np.float32), -1, roberts_x)
G_y = cv2.filter2D(test_gray.astype(np.float32), -1, roberts_y)
roberts = np.sqrt(G_x**2 + G_y**2)
roberts = np.clip(roberts, 0, 255).astype(np.uint8)

# Sobel算子（OpenCV内置）
sobel_x = cv2.Sobel(test_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(test_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

# Prewitt算子（手动实现）
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
P_x = cv2.filter2D(test_gray.astype(np.float32), -1, prewitt_x)
P_y = cv2.filter2D(test_gray.astype(np.float32), -1, prewitt_y)
prewitt = np.sqrt(P_x**2 + P_y**2)
prewitt = np.clip(prewitt, 0, 255).astype(np.uint8)

# 计算各方法的边缘强度
def edge_strength(img):
    return np.mean(img), np.max(img), np.sum(img > 50)

print("\n[边缘强度对比]")
print(f"{'算子':>10} | {'平均':>8} | {'最大':>8} | {'强边缘数':>10}")
print("-" * 45)
for name, img in [("Roberts", roberts), ("Sobel", sobel), ("Prewitt", prewitt)]:
    avg, mx, strong = edge_strength(img)
    print(f"{name:>10} | {avg:>8.1f} | {mx:>8.1f} | {strong:>10}")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(test_gray, cmap="gray")
axes[0, 0].set_title("原图（灰度）")
axes[0, 0].axis("off")

operators = [
    ("Roberts\n（2×2，对角差分）", roberts),
    ("Prewitt\n（3×3，简单平滑）", prewitt),
    ("Sobel\n（3×3，加权平滑[1,2,1]）", sobel),
]

for col, (title, img) in enumerate(operators, 1):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

# 放大一个区域看细节
crop_y, crop_x = 100, 80
crop_size = 80

for col, (title, img) in enumerate(operators, 1):
    crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    axes[1, col].imshow(crop, cmap="gray")
    axes[1, col].set_title(f"{title.split(chr(10))[0]}（局部放大）", fontsize=9)
    axes[1, col].axis("off")

axes[1, 0].imshow(test_gray[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 0].set_title("原图（局部放大）")
axes[1, 0].axis("off")

plt.suptitle("实验一：三种差分算子对比\n（Roberts最简单但噪声敏感，Sobel最常用）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("09_边缘检测基础/实验结果_算子对比.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存")
plt.show()

# 【实验现象1】Roberts的边缘最细（因为核最小），但噪声也最多
# 【实验现象2】Sobel和Prewitt的边缘较粗（因为核更大），但更稳定
# 【实验现象3】Sobel因为中心行权重[1,2,1]更大，边缘比Prewitt更"锐"

# ============================================================
# 实验二：噪声对边缘检测的影响 + 高斯平滑的作用
# ============================================================
print("\n" + "=" * 60)
print("实验二：噪声对边缘检测的影响 + 高斯平滑的作用")
print("=" * 60)

# 加高斯噪声
noisy = test_gray.copy()
noise = np.random.normal(0, 25, test_gray.shape).astype(np.int16)
noisy = np.clip(test_gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 无预处理直接Sobel
sobel_noisy = cv2.Sobel(noisy, cv2.CV_64F, 1, 0, ksize=3)
sobel_noisy = np.clip(sobel_noisy, 0, 255).astype(np.uint8)

# 先高斯平滑再Sobel
gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.5)
sobel_smooth = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
sobel_smooth = np.clip(sobel_smooth, 0, 255).astype(np.uint8)

# 先均值滤波再Sobel（对比）
mean_blur = cv2.blur(noisy, (5, 5))
sobel_mean = cv2.Sobel(mean_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_mean = np.clip(sobel_mean, 0, 255).astype(np.uint8)

# 计算边缘质量（用原图的边缘作为ground truth）
gt_edges = cv2.Sobel(test_gray, cv2.CV_64F, 1, 0, ksize=3)
gt_edges = np.clip(gt_edges, 0, 255).astype(np.uint8)

def edge_accuracy(edges, gt):
    """计算边缘检测的准确率（边缘像素的重合度）"""
    edges_binary = (edges > 50).astype(float)
    gt_binary = (gt > 50).astype(float)
    # F1-like score
    intersection = np.sum(edges_binary * gt_binary)
    precision = intersection / (np.sum(edges_binary) + 1e-6)
    recall = intersection / (np.sum(gt_binary) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

p1, r1, f1_1 = edge_accuracy(sobel_noisy, gt_edges)
p2, r2, f1_2 = edge_accuracy(sobel_smooth, gt_edges)
p3, r3, f1_3 = edge_accuracy(sobel_mean, gt_edges)

print(f"\n[边缘检测质量（F1分数，越高越好）]")
print(f"{'方法':>20} | {'Precision':>10} | {'Recall':>10} | {'F1':>8}")
print("-" * 55)
print(f"{'直接Sobel（无预处理）':>20} | {p1:>10.3f} | {r1:>10.3f} | {f1_1:>8.3f}")
print(f"{'高斯平滑+Sobel':>20} | {p2:>10.3f} | {r2:>10.3f} | {f1_2:>8.3f}")
print(f"{'均值平滑+Sobel':>20} | {p3:>10.3f} | {r3:>10.3f} | {f1_3:>8.3f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(test_gray, cmap="gray")
axes[0, 0].set_title("原图（无噪声）")
axes[0, 0].axis("off")

axes[0, 1].imshow(noisy, cmap="gray")
axes[0, 1].set_title("加高斯噪声（σ=25）")
axes[0, 1].axis("off")

axes[0, 2].imshow(gaussian, cmap="gray")
axes[0, 2].set_title("高斯平滑（σ=1.5）")
axes[0, 2].axis("off")

axes[1, 0].imshow(sobel_noisy, cmap="gray")
axes[1, 0].set_title(f"直接Sobel（F1={f1_1:.2f}⚠️噪声多）")
axes[1, 0].axis("off")

axes[1, 1].imshow(sobel_mean, cmap="gray")
axes[1, 1].set_title(f"均值平滑+Sobel（F1={f1_3:.2f}）")
axes[1, 1].axis("off")

axes[1, 2].imshow(sobel_smooth, cmap="gray")
axes[1, 2].set_title(f"高斯平滑+Sobel（F1={f1_2:.2f}✓）")
axes[1, 2].axis("off")

plt.suptitle("实验二：噪声→边缘检测变差→高斯平滑是解法\n（Sobel是差分算子，放大高频→放大噪声→需要先平滑）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("09_边缘检测基础/实验结果_噪声影响.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# 【实验现象1】直接对噪声图做Sobel，F1分数很低（噪声被当成边缘）
# 【实验现象2】高斯平滑后再Sobel，F1分数显著提升（高斯滤波有效压低噪声）
# 【实验现象3】高斯比均值效果好（因为高斯加权，边缘附近的平滑更少）

# ============================================================
# 实验三：阈值分割——固定阈值 / Otsu / 自适应
# ============================================================
print("\n" + "=" * 60)
print("实验三：固定阈值 vs Otsu vs 自适应阈值")
print("=" * 60)

# 构造两个场景：双峰分布和单峰分布
# 场景1：双峰（目标和背景分离）
bimodal = np.zeros((200, 400), dtype=np.uint8)
bimodal[:, :200] = 60 + np.random.normal(0, 8, (200, 200)).astype(np.int16)
bimodal[:, 200:] = 200 + np.random.normal(0, 8, (200, 200)).astype(np.int16)
bimodal = np.clip(bimodal, 0, 255).astype(np.uint8)

# 场景2：单峰（无明显目标和背景区分）
unimodal = np.zeros((200, 400), dtype=np.uint8)
unimodal[:] = 128 + np.random.normal(0, 25, (200, 400)).astype(np.int16)
unimodal = np.clip(unimodal, 0, 255).astype(np.uint8)

# 场景3：光照不均
uneven = np.zeros((200, 400), dtype=np.uint8)
for i in range(400):
    brightness = int(80 + i * 0.4)
    noise_val = np.random.normal(0, 10)
    unimodal_col = np.clip(brightness + noise_val, 0, 255).astype(np.uint8)
    uneven[:, i] = unimodal_col
# 加一些目标
uneven[80:120, 100:300] = np.clip(uneven[80:120, 100:300] + 80, 0, 255)

# 固定阈值
def fixed_threshold(img, T):
    _, binary = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return binary

# Otsu阈值
def otsu_threshold(img):
    T, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return T, binary

# 自适应阈值
def adaptive_threshold(img):
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    return binary

# 对三个场景应用三种阈值方法
scenarios = [
    ("双峰分布\n（目标和背景分离）", bimodal),
    ("单峰分布\n（无明显分割点）", unimodal),
    ("光照不均\n（左暗右亮）", uneven),
]

fig, axes = plt.subplots(len(scenarios), 5, figsize=(20, 12))

for row, (name, img) in enumerate(scenarios):
    # 原图
    axes[row, 0].imshow(img, cmap="gray")
    axes[row, 0].set_title(name, fontsize=10)
    axes[row, 0].axis("off")

    # 固定阈值（T=128）
    fixed = fixed_threshold(img, 128)
    axes[row, 1].imshow(fixed, cmap="gray")
    axes[row, 1].set_title("固定阈值 T=128", fontsize=9)
    axes[row, 1].axis("off")

    # Otsu
    if row < 2:  # Otsu不适合单峰，但仍然会返回一个T
        T_otsu, otsu = otsu_threshold(img)
        axes[row, 2].imshow(otsu, cmap="gray")
        axes[row, 2].set_title(f"Otsu T={T_otsu}", fontsize=9)
        axes[row, 2].axis("off")
    else:
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].set_title("Otsu失效\n（单峰分布）", fontsize=9)
        axes[row, 2].axis("off")

    # 自适应阈值
    adaptive = adaptive_threshold(img)
    axes[row, 3].imshow(adaptive, cmap="gray")
    axes[row, 3].set_title("自适应阈值\n(GAUSSIAN_C)", fontsize=9)
    axes[row, 3].axis("off")

    # 直方图
    axes[row, 4].hist(img.ravel(), bins=50, color="gray", alpha=0.7)
    axes[row, 4].axvline(x=128, color="b", linestyle="--", label="T=128")
    if row < 2:
        T_otsu, _ = otsu_threshold(img)
        axes[row, 4].axvline(x=T_otsu, color="r", linestyle="--", label=f"Otsu T={T_otsu:.0f}")
    axes[row, 4].set_title("灰度直方图", fontsize=9)
    axes[row, 4].legend(fontsize=7)

plt.suptitle("实验三：阈值分割方法对比\n（双峰→Otsu好 | 单峰→Otsu失效 | 光照不均→自适应阈值好）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("09_边缘检测基础/实验结果_阈值分割.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# 【实验现象1】双峰分布：Otsu找到的T接近两峰之间的谷底，分割效果好
# 【实验现象2】单峰分布：Otsu找到一个T，但分割结果几乎没有意义
# 【实验现象3】光照不均：固定阈值在整张图用一个T，左边过分割/右边欠分割

# 打印Otsu在单峰分布上的失效说明
T_uni = otsu_threshold(unimodal)[0]
print(f"\n[Otsu在单峰分布上失效]")
print(f"  单峰图像直方图是单峰的，Otsu遍历0-255找'最大化类间方差'的T")
print(f"  但单峰分布没有两个明显的类，这个T没有物理意义")
print(f"  Otsu返回的T={T_uni}，但用这个T做分割几乎没有效果")

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. Roberts/Prewitt/Sobel都是差分算子，差异在于核大小和权重设计")
print("2. 差分放大高频→放大噪声→边缘检测前必须先平滑（高斯>均值）")
print("3. 固定阈值：快速但需要人工调；Otsu：自动但需要双峰分布")
print("4. 自适应阈值：分区域各定T，适合光照不均的场景")
print("5. Otsu在单峰直方图上失效——这时候需要换思路（如基于熵的分割）")
