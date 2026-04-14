"""
第二章：图像的类型 - 实验演示
==================================
实验目标：
  1. 理解二值图、灰度图、RGB图在数值层面的区别
  2. 观察索引图的处理陷阱（调色板问题）
  3. 理解彩色→灰度转换的加权公式

实验准备：
  pip install opencv-python numpy matplotlib

运行：
  python demo_types.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 实验一：四种图像类型的数值结构对比
# ============================================================
print("=" * 60)
print("实验一：二值 / 灰度 / RGB / 索引 的数值结构")
print("=" * 60)

# 构造一张合成图：左半边黑(0)，右半边白(255)
canvas = np.zeros((200, 400), dtype=np.uint8)
canvas[:, 200:] = 255

# 【实验现象1】二值图像
binary = (canvas > 128).astype(np.uint8) * 255
# 二值图的像素只有0和255，不存在中间值
unique_vals = np.unique(binary)
print(f"[二值图] 唯一像素值: {unique_vals}")  # → [0, 255]
print(f"[二值图] 数据类型: {binary.dtype}, 通道数: {binary.ndim if len(binary.shape)==2 else binary.shape[2]}")
# 验证：二值图的"灰度值255"不等于灰度图的白，而是语义标签

# 【实验现象2】灰度图像
gray = canvas.copy()
unique_vals_gray = np.unique(gray)
print(f"\n[灰度图] 唯一像素值数量: {len(unique_vals_gray)}")
print(f"[灰度图] 前10个灰度值: {unique_vals_gray[:10]}")  # → 只有0和255（因为我们构造的就是二值灰度）
print(f"[灰度图] 灰度值范围: {gray.min()} ~ {gray.max()}")

# 构造真实灰度渐变图，观察灰度图能存的所有中间值
gradient = np.linspace(0, 255, 400, dtype=np.uint8)
gradient_img = np.tile(gradient, (100, 1))  # 100行，每行从0渐变到255
unique_gradient = np.unique(gradient_img)
print(f"\n[灰度渐变图] 唯一像素值数量: {len(unique_gradient)}")  # → 400（每个像素都是不同的值）

# 【实验现象3】RGB彩色图像
rgb_canvas = np.zeros((200, 400, 3), dtype=np.uint8)
rgb_canvas[:, :200] = [30, 30, 30]    # 左半边：暗灰
rgb_canvas[:, 200:] = [180, 50, 50]   # 右半边：红棕色
print(f"\n[RGB图] 形状: {rgb_canvas.shape} (高×宽×通道)")
print(f"[RGB图] 左半边像素值: {rgb_canvas[0, 0]}")   # → [30, 30, 30]
print(f"[RGB图] 右半边像素值: {rgb_canvas[0, 201]}")  # → [180, 50, 50]
# 【关键观察】三个通道数值耦合——如果光照变强，三个通道同比例增大

# 光照模拟：整体亮度+50
brightened = np.clip(rgb_canvas.astype(int) + 50, 0, 255).astype(np.uint8)
print(f"\n[RGB图+50亮度] 左半边: {brightened[0, 0]}")  # → [80, 80, 80]（原来是[30,30,30]）
ratio_r = brightened[0, 201, 0] / rgb_canvas[0, 201, 0]
ratio_g = brightened[0, 201, 1] / rgb_canvas[0, 201, 1]
ratio_b = brightened[0, 201, 2] / rgb_canvas[0, 201, 2]
print(f"[RGB图+50亮度] 右半边比值: R={ratio_r:.2f}, G={ratio_g:.2f}, B={ratio_b:.2f}")
# 【实验现象】三个通道同比例放大 → RGB值变了，但"红棕色"的色调没变

# ============================================================
# 实验二：彩色→灰度转换，公式差异导致结果不同
# ============================================================
print("\n" + "=" * 60)
print("实验二：灰度转换公式差异（OpenCV vs PIL vs MATLAB）")
print("=" * 60)

# OpenCV BGR: 0.114*B + 0.587*G + 0.299*R
# Python PIL: 0.299*R + 0.587*G + 0.114*B（注意顺序！）
# MATLAB:     0.2989*R + 0.5870*G + 0.1140*B

test_rgb = np.array([[[200, 100, 50]]], dtype=np.uint8)  # 模拟一个RGB像素：R=200, G=100, B=50

# OpenCV公式（针对BGR顺序）
opencv_gray = int(0.114 * test_rgb[0, 0, 0] + 0.587 * test_rgb[0, 0, 1] + 0.299 * test_rgb[0, 0, 2])
# PIL公式（针对RGB顺序）
pil_gray = int(0.299 * test_rgb[0, 0, 2] + 0.587 * test_rgb[0, 0, 1] + 0.114 * test_rgb[0, 0, 0])

print(f"[原始像素] B={test_rgb[0,0,0]}, G={test_rgb[0,0,1]}, R={test_rgb[0,0,2]}")
print(f"[OpenCV公式] 灰度值 = 0.114*B + 0.587*G + 0.299*R = {opencv_gray}")
print(f"[PIL公式]    灰度值 = 0.299*R + 0.587*G + 0.114*B = {pil_gray}")
print(f"[差异] OpenCV和PIL的灰度值相差: {abs(opencv_gray - pil_gray)}")
# 【实验现象】同一个像素，两种公式算出来的灰度值不同！
# 这在需要精确像素值对比的场景（医疗、测量）里是潜在bug来源

# 验证：实际用OpenCV读取一张真实图片时，
# cvtColor默认公式是 BGR→Gray，系数和上面一致

# ============================================================
# 实验三：索引图像的调色板陷阱（用BMP演示）
# ============================================================
print("\n" + "=" * 60)
print("实验三：索引图像调色板陷阱（概念演示）")
print("=" * 60)

# 创建一张假索引图：像素值是1, 2, 3..., 对应调色板里的RGB
fake_indexed = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)

# 构造调色板：每行是一个RGB值
# 假设：索引1=红, 2=绿, 3=蓝, 4=黄, 5=品红, 6=青, 7=白, 8=黑, 9=灰
palette = np.zeros((256, 3), dtype=np.uint8)
palette[1] = [255, 0, 0]     # 红
palette[2] = [0, 255, 0]     # 绿
palette[3] = [0, 0, 255]     # 蓝
palette[4] = [255, 255, 0]   # 黄
palette[5] = [255, 0, 255]   # 品红
palette[6] = [0, 255, 255]   # 青
palette[7] = [255, 255, 255] # 白
palette[8] = [0, 0, 0]       # 黑
palette[9] = [128, 128, 128] # 灰

# 错误做法：直接把索引值当灰度值处理
wrong_gray = fake_indexed.astype(np.uint8)  # 把索引值1,2,3...当灰度值
print(f"[错误做法] 把索引值直接当灰度: {wrong_gray.tolist()}")
print(f"→ 索引1被当成灰度值1（几乎是纯黑），但实际它应该是红色(RGB=[255,0,0])")

# 正确做法：通过调色板查表得到真实颜色
correct_rgb = np.zeros((*fake_indexed.shape, 3), dtype=np.uint8)
for i in range(fake_indexed.shape[0]):
    for j in range(fake_indexed.shape[1]):
        idx = fake_indexed[i, j]
        correct_rgb[i, j] = palette[idx]
print(f"[正确做法] 通过调色板查表:")
print(f"索引1→{correct_rgb[0,0]}（红）")
print(f"索引2→{correct_rgb[0,1]}（绿）")
print(f"索引3→{correct_rgb[0,2]}（蓝）")
# 【实验现象】索引值1,2,3在错误做法里是"灰度1,2,3"，在正确做法里是"红,绿,蓝"

# ============================================================
# 实验四：可视化四种图像类型的视觉效果差异
# ============================================================
print("\n" + "=" * 60)
print("实验四：可视化四种图像类型（请查看弹出的窗口）")
print("=" * 60)

# 构造一张测试图：渐变+色块+噪声
test_img = np.zeros((300, 600, 3), dtype=np.uint8)
# 左上：红色渐变
test_img[:100, :200, 2] = np.linspace(0, 255, 200, dtype=np.uint8)[np.newaxis, :]
test_img[:100, :200, 0] = 200
# 右上：绿色
test_img[:100, 200:, 1] = 200
# 左下：蓝色
test_img[100:, :200, 2] = 200
# 右下：白色噪声
test_img[100:, 200:] = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

# 转灰度
gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# 转二值（固定阈值130）
_, binary_test = cv2.threshold(gray_test, 130, 255, cv2.THRESH_BINARY)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("RGB彩色图")
axes[0].axis("off")

axes[1].imshow(gray_test, cmap="gray")
axes[1].set_title("灰度图")
axes[1].axis("off")

axes[2].imshow(binary_test, cmap="gray")
axes[2].set_title("二值图 (T=130)")
axes[2].axis("off")

# 灰度直方图
axes[3].hist(gray_test.ravel(), bins=50, color="gray")
axes[3].set_title("灰度直方图（用于阈值选择）")
axes[3].set_xlabel("灰度值")
axes[3].set_ylabel("像素个数")

plt.suptitle("四种图像类型对比实验", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_图像类型/实验结果_四类图像对比.png", dpi=150, bbox_inches="tight")
print("[保存] 实验结果图已保存到 02_图像类型/实验结果_四类图像对比.png")
plt.show()

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. 二值图只有0/255，灰度图有0-255所有值，RGB图是三通道叠回")
print("2. RGB三通道耦合 → 光照变化时RGB值全变，但色调不变")
print("3. 灰度转换公式不同结果不同（OpenCV vs PIL差10+）")
print("4. 索引图必须查调色板，否则索引值=灰度值=错误颜色")
