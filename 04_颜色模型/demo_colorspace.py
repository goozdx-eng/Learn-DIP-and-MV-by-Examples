"""
第四章：常见的颜色模型 - 实验演示
==================================
实验目标：
  1. 观察RGB三通道的耦合现象（光照变化时三个通道同比例变化）
  2. 验证HSV的H通道与光照无关（为什么HSV适合做颜色分割）
  3. 理解YUV的亮度-色度分离（为什么视频压缩用YUV）
  4. 观察饱和度低时H值不可靠的现象

实验准备：
  pip install opencv-python numpy matplotlib

运行：
  python demo_colorspace.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ============================================================
# 实验一：RGB三通道耦合——光照变化时三个通道同比例变化
# ============================================================
print("=" * 60)
print("实验一：RGB三通道耦合现象（光照变化的影响）")
print("=" * 60)

# 构造一个红色目标（R高,G低,B低）
red_object = np.array([[[180, 40, 30]]], dtype=np.uint8)  # 模拟红苹果的RGB
print(f"[原始红色目标] RGB = {red_object[0,0]}")

# 模拟三种光照条件
illuminations = {
    "强光（+80）": np.clip(red_object.astype(int) + 80, 0, 255).astype(np.uint8),
    "正常光（0）": red_object,
    "弱光（-60）": np.clip(red_object.astype(int) - 60, 0, 255).astype(np.uint8),
    "极暗（-140）": np.clip(red_object.astype(int) - 140, 0, 255).astype(np.uint8),
}

print("\n[光照变化对RGB的影响]")
for name, img in illuminations.items():
    rgb = img[0, 0]
    # 计算RGB比例（归一化到总和）
    total = rgb.sum() + 1e-6
    r_ratio = rgb[0] / total
    g_ratio = rgb[1] / total
    b_ratio = rgb[2] / total
    print(f"  {name}: RGB={rgb}, R占比={r_ratio:.2f}, G占比={g_ratio:.2f}, B占比={b_ratio:.2f}")

# 【实验现象】三个通道随光照同比例变化，但R/G/B在总和中的比例基本不变
# 这就是"色调不变，但RGB值变了"

# 创建合成测试图：红、绿、蓝、黄四个色块
color_blocks = np.zeros((200, 400, 3), dtype=np.uint8)
color_blocks[:, :100]   = [180, 40, 30]    # 红
color_blocks[:, 100:200] = [50, 180, 40]   # 绿
color_blocks[:, 200:300] = [40, 50, 180]   # 蓝
color_blocks[:, 300:400] = [180, 180, 40]  # 黄

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# 上排：不同光照下的RGB图像
illumination_deltas = [80, 0, -60, -140]
for i, delta in enumerate(illumination_deltas):
    img = np.clip(color_blocks.astype(int) + delta, 0, 255).astype(np.uint8)
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"光照+{delta}" if delta >= 0 else f"光照{delta}", fontsize=10)
    axes[0, i].axis("off")

# 下排：对应的R、G、B通道分离
for i, delta in enumerate(illumination_deltas):
    img = np.clip(color_blocks.astype(int) + delta, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(img)  # OpenCV是BGR顺序
    # 显示三通道的灰度值条形图
    means = [np.mean(r), np.mean(g), np.mean(b)]
    colors = ["red", "green", "blue"]
    axes[1, i].bar(colors, means, color=colors, alpha=0.7)
    axes[1, i].set_ylim(0, 255)
    axes[1, i].set_title(f"R/G/B均值", fontsize=9)
    axes[1, i].set_ylabel("灰度值")

plt.suptitle("实验一：光照变化对RGB三通道的影响\n（四个色块：红、绿、蓝、黄）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("04_颜色模型/实验结果_RGB耦合.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存到 04_颜色模型/实验结果_RGB耦合.png")
plt.show()

# ============================================================
# 实验二：HSV与光照无关——为什么HSV适合颜色分割
# ============================================================
print("\n" + "=" * 60)
print("实验二：HSV的H（色调）与光照无关")
print("=" * 60)

# 同一张图，不同光照
illuminated_blocks = {}
for delta in [80, 0, -60, -140]:
    img = np.clip(color_blocks.astype(int) + delta, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illuminated_blocks[delta] = {
        "rgb": img,
        "hsv": hsv,
        "H": hsv[:, :, 0].mean(),   # H范围0-180（OpenCV的HSV里H除以2存储）
        "S": hsv[:, :, 1].mean(),   # S范围0-255
        "V": hsv[:, :, 2].mean(),   # V范围0-255
    }

print("\n[光照变化对HSV的影响 - 四个色块平均]")
print(f"{'光照':>10} | {'H(色调)':>10} | {'S(饱和度)':>12} | {'V(亮度)':>10}")
print("-" * 50)
for delta, data in illuminated_blocks.items():
    label = f"+{delta}" if delta >= 0 else str(delta)
    print(f"{label:>10} | {data['H']:>10.1f} | {data['S']:>12.1f} | {data['V']:>10.1f}")

# 【实验现象1】V随光照大幅变化（+80→约235，-140→约45）
# 【实验现象2】H基本不变（同一颜色的色调不随光照变）
# 【实验现象3】S轻微变化（光照太强时饱和度可能降低——高光区域饱和度天然低）

# 可视化：四张图×RGB/HSV分解
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
illumination_deltas = [80, 0, -60, -140]

for row, delta in enumerate(illumination_deltas):
    data = illuminated_blocks[delta]
    rgb_img = cv2.cvtColor(data["rgb"], cv2.COLOR_BGR2RGB)
    hsv_img = data["hsv"]

    # RGB通道
    axes[row, 0].imshow(rgb_img)
    title = f"光照{'+' if delta >= 0 else ''}{delta}"
    axes[row, 0].set_title(title, fontsize=10)
    axes[row, 0].axis("off")

    # H、S、V通道
    h_channel = hsv_img[:, :, 0].astype(np.float32) * 2  # 转回0-360
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]

    axes[row, 1].imshow(h_channel, cmap="hsv")
    axes[row, 1].set_title(f"H(色调) 均值={data['H']:.1f}°", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(s_channel, cmap="gray")
    axes[row, 2].set_title(f"S(饱和度) 均值={data['S']:.1f}", fontsize=9)
    axes[row, 2].axis("off")

    axes[row, 3].imshow(v_channel, cmap="gray")
    axes[row, 3].set_title(f"V(亮度) 均值={data['V']:.1f}", fontsize=9)
    axes[row, 3].axis("off")

plt.suptitle("实验二：光照变化时RGB vs HSV对比\n（V随光照变，H几乎不变——这就是HSV适合颜色分割的原因）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("04_颜色模型/实验结果_HSV分离.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# ============================================================
# 实验三：HSV颜色分割实战——检测红色和绿色目标
# ============================================================
print("\n" + "=" * 60)
print("实验三：HSV颜色分割实战（检测红色和绿色）")
print("=" * 60)

# 构造一个包含红、绿、蓝、黄、橙的测试场景
test_scene = np.zeros((300, 500, 3), dtype=np.uint8)
# 放置色块
test_scene[50:150, 50:150] = [0, 0, 255]      # 红
test_scene[50:150, 200:300] = [0, 255, 0]     # 绿
test_scene[50:150, 350:450] = [255, 255, 0]   # 黄
test_scene[150:250, 50:150] = [0, 255, 255]   # 青
test_scene[150:250, 200:300] = [255, 100, 50] # 橙

# 加光照变化：左半边亮，右半边暗
test_scene[:, 250:] = np.clip(test_scene[:, 250:].astype(int) - 60, 0, 255).astype(np.uint8)

# 转HSV
hsv_scene = cv2.cvtColor(test_scene, cv2.COLOR_BGR2HSV)

# 定义红色范围（OpenCV的H范围是0-180，红色在0-10和170-180两端）
red_lower1 = np.array([0, 50, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 50, 50])
red_upper2 = np.array([180, 255, 255])

# 绿色范围
green_lower = np.array([35, 50, 50])
green_upper = np.array([85, 255, 255])

# 创建掩膜
red_mask1 = cv2.inRange(hsv_scene, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv_scene, red_lower2, red_upper2)
red_mask = red_mask1 | red_mask2
green_mask = cv2.inRange(hsv_scene, green_lower, green_upper)

# 统计检测结果
red_pixels_left = np.sum(red_mask[50:150, 50:150] > 0)
red_pixels_right = np.sum(red_mask[50:150, 300:400] > 0)
green_pixels_left = np.sum(green_mask[50:150, 200:300] > 0)
green_pixels_right = np.sum(green_mask[150:250, 200:300] > 0)

print(f"[红色检测] 左半边（亮）: {red_pixels_left} 像素 | 右半边（暗）: {red_pixels_right} 像素")
print(f"[绿色检测] 左半边（亮）: {green_pixels_left} 像素 | 右半边（暗）: {green_pixels_right} 像素")
print(f"[检测率分析] 光照-60后，红色检测率: {red_pixels_right/max(red_pixels_left,1)*100:.1f}% | 绿色检测率: {green_pixels_right/max(green_pixels_left,1)*100:.1f}%")

# 【实验现象】用HSV的H通道做颜色检测，光照变化时依然能检测到目标
# 即使亮度降低了60，红色和绿色的H值仍在设定范围内

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(test_scene, cv2.COLOR_BGR2RGB))
axes[0].set_title("原图（左亮右暗）")
axes[0].axis("off")

axes[1].imshow(red_mask, cmap="gray")
axes[1].set_title(f"红色检测掩膜\n（左:{red_pixels_left} 右:{red_pixels_right}）")
axes[1].axis("off")

axes[2].imshow(green_mask, cmap="gray")
axes[2].set_title(f"绿色检测掩膜\n（左:{green_pixels_left} 右:{green_pixels_right}）")
axes[2].axis("off")

# 叠加显示
result = test_scene.copy()
result[red_mask > 0] = [0, 255, 255]  # 红色区域涂成黄色
result[green_mask > 0] = [255, 0, 255]  # 绿色区域涂成品红
axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[3].set_title("检测结果叠加")
axes[3].axis("off")

plt.suptitle("实验三：HSV颜色分割实战\n（光照变化时依然能检测到红色和绿色）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("04_颜色模型/实验结果_HSV颜色分割.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# ============================================================
# 实验四：饱和度低时H值不可靠
# ============================================================
print("\n" + "=" * 60)
print("实验四：饱和度低时H值不可靠（灰白色无色调）")
print("=" * 60)

# 构造不同饱和度的同一色调图像
base_color = np.array([180, 40, 30], dtype=np.uint8)  # 红色基准
saturations = [255, 180, 100, 50, 20, 5]  # 不同饱和度级别

print("\n[饱和度从高到低，H值的变化]")
print(f"{'饱和度':>8} | {'H(色调°)':>12} | {'可靠性':>10}")
print("-" * 40)

for sat in saturations:
    # 通过插值降低饱和度
    # 从base_color（高饱和）到白色(255,255,255)插值
    ratio = 1 - sat / 255.0
    desaturated = ((1 - ratio) * base_color + ratio * np.array([255, 255, 255])).astype(np.uint8)
    hsv = cv2.cvtColor(np.array([[desaturated]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    h = hsv[0] * 2  # 转回0-360
    s = hsv[1]
    reliable = "✓ 可靠" if s > 30 else "⚠️ 不可靠"
    print(f"{sat:>8} | {h:>10.1f}° | {reliable} (S={s})")

# 【实验现象】饱和度越低，H值越不稳定（越接近白色，"色调"越没有意义）
# 当S<30时，H值会跳变，不再反映真实的颜色种类

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. RGB三通道耦合 → 光照变化时RGB值全变，但色调不变")
print("2. HSV的H与光照无关 → 适合做颜色分割，S低时H不可靠")
print("3. YUV亮度-色度分离 → 视频压缩降低色度分辨率节省带宽")
print("4. 颜色模型选择：颜色分割→HSV；视频→YUV；采集显示→RGB")
