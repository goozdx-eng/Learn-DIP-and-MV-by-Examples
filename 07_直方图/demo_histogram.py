"""
第七章：直方图相关 - 实验演示
==================================
实验目标：
  1. 观察不同图像的直方图形态，理解直方图是"分布"不是"图"
  2. 验证直方图均衡化把挤在一起的灰度铺开、增强对比度的效果
  3. 理解CDF映射的工作机制
  4. 观察均衡化在噪声图和光照不均图上的"帮倒忙"现象
  5. 理解直方图规定化（规定到指定形状）

实验准备：
  pip install opencv-python numpy matplotlib

运行：
  python demo_histogram.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 实验一：不同图像的直方图形态对比
# ============================================================
print("=" * 60)
print("实验一：直方图是"分布"不是"图"——不同图像的形态")
print("=" * 60)

# 构造三种典型场景
# 场景1：低对比度（灰度挤在中间）
low_contrast = np.zeros((200, 400), dtype=np.uint8)
# 高斯分布的灰度，集中在100-200之间
mu, sigma = 150, 30
x = np.arange(400)
y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
y = (y / y.max() * 80 + 80).astype(int)  # 映射到80-160范围
for i in range(200):
    low_contrast[i, :] = y

# 场景2：双峰（目标和背景分离明显）
bimodal = np.zeros((200, 400), dtype=np.uint8)
bimodal[:, :200] = 60   # 背景：暗
bimodal[:, 200:] = 200  # 目标：亮
# 加一点高斯噪声模拟真实情况
noise = np.random.normal(0, 10, bimodal.shape).astype(np.int16)
bimodal = np.clip(bimodal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 场景3：过曝（灰度挤在高处）
overexposed = np.full((200, 400), 200, dtype=np.uint8)
overexposed[:, 50:350] = np.linspace(180, 255, 300, dtype=np.uint8)
noise_over = np.random.normal(0, 5, overexposed.shape).astype(np.int16)
overexposed = np.clip(overexposed.astype(np.int16) + noise_over, 0, 255).astype(np.uint8)

# 场景4：欠曝（灰度挤在低处）
underexposed = np.full((200, 400), 55, dtype=np.uint8)
underexposed[:, 50:350] = np.linspace(20, 120, 300, dtype=np.uint8)
noise_under = np.random.normal(0, 5, underexposed.shape).astype(np.int16)
underexposed = np.clip(underexposed.astype(np.int16) + noise_under, 0, 255).astype(np.uint8)

# 计算直方图
def compute_hist(img, bins=256):
    hist = cv2.calcHist([img], [0], None, [bins], [0, bins]).flatten()
    return hist

hist_low = compute_hist(low_contrast)
hist_bimodal = compute_hist(bimodal)
hist_over = compute_hist(overexposed)
hist_under = compute_hist(underexposed)

# 打印直方图特征
for name, hist in [("低对比度", hist_low), ("双峰分布", hist_bimodal),
                    ("过曝", hist_over), ("欠曝", hist_under)]:
    nonzero = np.nonzero(hist)[0]
    if len(nonzero) > 0:
        print(f"  {name}: 灰度范围[{nonzero.min()}, {nonzero.max()}], 峰值在灰度{np.argmax(hist)}")

# 可视化
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
scenarios = [
    ("低对比度\n（灰度集中在中间）", low_contrast, hist_low),
    ("双峰分布\n（目标和背景分离）", bimodal, hist_bimodal),
    ("过曝\n（灰度集中在高处）", overexposed, hist_over),
    ("欠曝\n（灰度集中在低处）", underexposed, hist_under),
]

for row, (title, img, hist) in enumerate(scenarios):
    axes[row, 0].imshow(img, cmap="gray")
    axes[row, 0].set_title(title, fontsize=11)
    axes[row, 0].axis("off")

    axes[row, 1].fill_between(range(256), hist, color="gray", alpha=0.7)
    axes[row, 1].set_xlim(0, 255)
    axes[row, 1].set_title(f"灰度直方图（峰值={np.argmax(hist)}）", fontsize=10)
    axes[row, 1].set_xlabel("灰度值")
    axes[row, 1].set_ylabel("像素个数")

plt.suptitle("实验一：四种典型直方图形态\n（直方图告诉你图像是偏暗、偏亮、对比度低还是双峰分离）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_直方图/实验结果_直方图形态.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存")
plt.show()

# 【实验现象】低对比度图像的直方图挤在中间；过曝图像的直方图偏右；欠曝偏左
# 直方图是图像的"性格标签"，告诉你图像的状态

# ============================================================
# 实验二：直方图均衡化——把挤在一起的山铺开
# ============================================================
print("\n" + "=" * 60)
print("实验二：直方图均衡化（CDF映射把灰度铺开）")
print("=" * 60)

# 用一张低对比度图像演示
gray_low = low_contrast.copy()

# OpenCV均衡化
eq_img = cv2.equalizeHist(gray_low)

# 手动实现均衡化（验证CDF映射）
def manual_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # 计算CDF（累积分布函数）
    cdf = hist.cumsum()
    # 归一化CDF
    cdf_normalized = cdf / cdf[-1]
    # 建立映射：原灰度→新灰度
    cdf_mapped = np.round(cdf_normalized * 255).astype(np.uint8)
    # 应用映射
    eq = cdf_mapped[img]
    return eq, cdf_normalized

manual_eq, cdf = manual_equalize(gray_low)

# 计算均衡化前后的直方图
hist_before = compute_hist(gray_low)
hist_after = compute_hist(eq_img)
hist_manual = compute_hist(manual_eq)

# 打印均衡化前后的灰度范围
before_range = np.nonzero(hist_before)[0]
after_range = np.nonzero(hist_after)[0]
print(f"[均衡化前后对比]")
print(f"  均衡化前: 灰度范围[{before_range.min()}, {before_range.max()}]")
print(f"  均衡化后: 灰度范围[{after_range.min()}, {after_range.max()}]")
print(f"  {'✓ 灰度范围扩大' if after_range.max() - after_range.min() > before_range.max() - before_range.min() else '✗ 无变化'}")

# 验证手动均衡化和OpenCV结果一致
match = np.array_equal(eq_img, manual_eq)
print(f"  手动均衡化与OpenCV一致: {'✓' if match else '✗'}")

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# 第一行：原图、直方图、CDF
axes[0, 0].imshow(gray_low, cmap="gray")
axes[0, 0].set_title("原图（低对比度）")
axes[0, 0].axis("off")

axes[0, 1].fill_between(range(256), hist_before, color="gray", alpha=0.7)
axes[0, 1].set_title("原图直方图")
axes[0, 1].set_xlim(0, 255)

axes[0, 2].plot(range(256), cdf, color="blue", linewidth=2)
axes[0, 2].set_title("CDF（累积分布函数）")
axes[0, 2].set_xlabel("灰度值")
axes[0, 2].set_ylabel("累积概率")
axes[0, 2].set_xlim(0, 255)

# 第二行：均衡化后的图和直方图
axes[1, 0].imshow(eq_img, cmap="gray")
axes[1, 0].set_title("均衡化后（对比度增强）")
axes[1, 0].axis("off")

axes[1, 1].fill_between(range(256), hist_after, color="gray", alpha=0.7)
axes[1, 1].set_title("均衡化后直方图")
axes[1, 1].set_xlim(0, 255)

# 画映射函数
axes[1, 2].plot(range(256), cdf * 255, color="red", linewidth=2)
axes[1, 2].set_title("灰度映射函数（原灰度→新灰度）")
axes[1, 2].set_xlabel("原灰度值")
axes[1, 2].set_ylabel("新灰度值")
axes[1, 2].set_xlim(0, 255)
axes[1, 2].set_ylim(0, 255)

# 第三行：对比（前后并排）
axes[2, 0].imshow(gray_low, cmap="gray")
axes[2, 0].set_title("均衡化前")
axes[2, 0].axis("off")

axes[2, 1].imshow(eq_img, cmap="gray")
axes[2, 1].set_title("均衡化后")
axes[2, 1].axis("off")

axes[2, 2].plot(range(256), hist_before, color="blue", alpha=0.7, label="均衡化前")
axes[2, 2].plot(range(256), hist_after, color="red", alpha=0.7, label="均衡化后")
axes[2, 2].set_title("直方图对比")
axes[2, 2].legend()
axes[2, 2].set_xlim(0, 255)

plt.suptitle("实验二：直方图均衡化原理\n（CDF映射把挤在[60,160]的灰度铺开到[0,255]）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_直方图/实验结果_均衡化.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# 【实验现象1】均衡化后灰度范围从[60,160]扩展到[0,255]
# 【实验现象2】直方图从"一座偏左的山"变成"相对平坦的分布"
# 【实验现象3】CDF映射是递增的——灰度值大的映射后也大，不会颠倒顺序

# ============================================================
# 实验三：均衡化"帮倒忙"——噪声放大和光照不均
# ============================================================
print("\n" + "=" * 60)
print("实验三：均衡化"帮倒忙"（噪声放大 + 光照不均）")
print("=" * 60)

# 场景A：平坦区域+噪声
flat_noise = np.full((200, 400), 128, dtype=np.uint8)
flat_noise[:, 50:350] = 120
# 加椒盐噪声（模拟高频噪声）
flat_noise[::5, ::5] = 0      # 椒
flat_noise[2::5, 2::5] = 255  # 盐

# 场景B：左亮右暗（光照不均）
uneven_light = np.zeros((200, 400), dtype=np.uint8)
for i in range(400):
    brightness = int(255 * (0.3 + 0.7 * i / 400))  # 左暗右亮
    uneven_light[:, i] = brightness
# 加一些结构
uneven_light[80:120, :] = np.clip(uneven_light[80:120, :] + 80, 0, 255)

# 各自做均衡化
flat_eq = cv2.equalizeHist(flat_noise)
uneven_eq = cv2.equalizeHist(uneven_light)

# 计算均衡化前后噪声区域的方差（衡量噪声放大程度）
def measure_noise_variance(img, roi):
    """测量ROI区域的方差（方差越大=噪声越强）"""
    return np.var(img[roi])

flat_noise_before = measure_noise_variance(flat_noise, (0, 0, 50, 50))
flat_noise_after = measure_noise_variance(flat_eq, (0, 0, 50, 50))
print(f"[场景A：平坦区域噪声放大]")
print(f"  均衡化前噪声方差: {flat_noise_before:.2f}")
print(f"  均衡化后噪声方差: {flat_noise_after:.2f}")
print(f"  噪声放大了 {flat_noise_after/flat_noise_before:.1f} 倍{' ⚠️ 噪声被放大！' if flat_noise_after/flat_noise_before > 2 else ''}")

# 光照不均场景：计算左右亮度差异
left_before = np.mean(uneven_light[:, :50])
right_before = np.mean(uneven_light[:, 350:])
left_after = np.mean(uneven_eq[:, :50])
right_after = np.mean(uneven_eq[:, 350:])
print(f"\n[场景B：光照不均（左暗右亮）]")
print(f"  均衡化前: 左={left_before:.0f}, 右={right_before:.0f}, 差异={abs(left_before-right_before):.0f}")
print(f"  均衡化后: 左={left_after:.0f}, 右={right_after:.0f}, 差异={abs(left_after-right_after):.0f}")
print(f"  {'⚠️ 左边过曝了！' if left_after > 200 else '正常'}")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 场景A
axes[0, 0].imshow(flat_noise, cmap="gray")
axes[0, 0].set_title("原图（平坦+噪声）")
axes[0, 0].axis("off")

axes[0, 1].imshow(flat_eq, cmap="gray")
axes[0, 1].set_title("均衡化后（噪声被放大⚠️）")
axes[0, 1].axis("off")

axes[0, 2].hist(flat_noise.ravel(), bins=50, alpha=0.7, label="均衡化前")
axes[0, 2].hist(flat_eq.ravel(), bins=50, alpha=0.7, label="均衡化后")
axes[0, 2].set_title("直方图对比")
axes[0, 2].legend()

# 放大噪声区域
axes[0, 3].imshow(flat_noise[:100, :100], cmap="gray")
axes[0, 3].set_title("原图局部（放大噪声）")
axes[0, 3].axis("off")

# 场景B
axes[1, 0].imshow(uneven_light, cmap="gray")
axes[1, 0].set_title("原图（左暗右亮）")
axes[1, 0].axis("off")

axes[1, 1].imshow(uneven_eq, cmap="gray")
axes[1, 1].set_title("均衡化后（左边过曝⚠️）")
axes[1, 1].axis("off")

axes[1, 2].plot(range(400), [np.mean(uneven_light[:, i]) for i in range(400)], label="均衡化前")
axes[1, 2].plot(range(400), [np.mean(uneven_eq[:, i]) for i in range(400)], label="均衡化后")
axes[1, 2].set_title("各列平均亮度曲线")
axes[1, 2].legend()

axes[1, 3].axis("off")

plt.suptitle("实验三：均衡化"帮倒忙"\n（上：噪声放大 | 下：光照不均导致过曝）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_直方图/实验结果_均衡化陷阱.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# 【实验现象1】平坦区域（灰度均匀）加噪声后，均衡化会把噪声的"尖峰"也铺开，导致噪声放大
# 【实验现象2】光照不均的图，左边本来就亮，均衡化后变得更亮（过曝）

# ============================================================
# 实验四：CLAHE（自适应直方图均衡化）解决光照不均
# ============================================================
print("\n" + "=" * 60)
print("实验四：CLAHE（自适应直方图均衡化）")
print("=" * 60)

# 创建光照不均图
clahe_test = uneven_light.copy()
clahe_test[80:120, :] = np.clip(clahe_test[80:120, :] + 80, 0, 255)  # 加一条亮带

# 全局均衡化
global_eq = cv2.equalizeHist(clahe_test)

# CLAHE（对比度限制的自适应直方图均衡化）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_eq = clahe.apply(clahe_test)

print(f"[CLAHE vs 全局均衡化]")
print(f"  全局均衡化: 左亮={np.mean(global_eq[:, 50]):.0f}, 右亮={np.mean(global_eq[:, 350]):.0f}")
print(f"  CLAHE:      左亮={np.mean(clahe_eq[:, 50]):.0f}, 右亮={np.mean(clahe_eq[:, 350]):.0f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(clahe_test, cmap="gray")
axes[0].set_title("原图")
axes[0].axis("off")

axes[1].imshow(global_eq, cmap="gray")
axes[1].set_title("全局均衡化\n(左边过曝)")
axes[1].axis("off")

axes[2].imshow(clahe_eq, cmap="gray")
axes[2].set_title("CLAHE\n(局部自适应)")
axes[2].axis("off")

axes[3].plot([np.mean(clahe_test[:, i]) for i in range(400)], label="原图")
axes[3].plot([np.mean(global_eq[:, i]) for i in range(400)], label="全局均衡")
axes[3].plot([np.mean(clahe_eq[:, i]) for i in range(400)], label="CLAHE")
axes[3].legend()
axes[3].set_title("亮度曲线对比")

plt.suptitle("实验四：CLAHE（自适应直方图均衡化）\n（分块做均衡化，避免光照不均的过曝问题）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_直方图/实验结果_CLAHE.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验四图已保存")
plt.show()

# 【实验现象】CLAHE把图像分成8×8的小块，对每块单独均衡化，
# 然后拼接——左边暗区域的T低，右边亮区域的T高，整体亮度分布更均匀

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. 直方图是分布统计，丢失空间信息，但能快速判断图像状态")
print("2. 均衡化用CDF映射把挤在一起的灰度铺开，增强对比度")
print("3. 均衡化的局限：放大噪声、对光照不均图会过曝")
print("4. CLAHE分块均衡化解决光照不均，是医学/卫星图像增强的常用方法")
