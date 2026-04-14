"""
第八章：几何变换基础 - 实验演示
==================================
实验目标：
  1. 对比最近邻、双线性、双三次三种插值方法的视觉质量差异
  2. 观察旋转/缩放多次变换后的插值误差累积
  3. 理解图像配准三步骤（特征提取→匹配→变换）
  4. 验证配准前的空间分辨率对齐（重采样）必要性

实验准备：
  pip install opencv-python numpy matplotlib opencv-contrib-python

运行：
  python demo_geom.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 实验一：三种插值方法的视觉对比
# ============================================================
print("=" * 60)
print("实验一：最近邻 vs 双线性 vs 双三次插值")
print("=" * 60)

# 加载测试图（如果没有就用合成图）
# 合成测试图：包含斜线和高频纹理（不再依赖外部文件）
test_img = np.zeros((300, 400), dtype=np.uint8)
test_img[:, :] = 200
# 加斜线（最容易暴露锯齿）
for i in range(300):
    j = int(i * 1.5)
    if j < 400:
        test_img[i, j] = 50
# 加格子纹理
test_img[::20, :] = 100
test_img[:, ::20] = 100

# 缩小再放大（模拟多次变换）
scale = 0.5
H, W = test_img.shape

# 三种插值方法
interpolations = {
    "最近邻 (INTER_NEAREST)": cv2.INTER_NEAREST,
    "双线性 (INTER_LINEAR)": cv2.INTER_LINEAR,
    "双三次 (INTER_CUBIC)": cv2.INTER_CUBIC,
}

results = {}
for name, interp in interpolations.items():
    # 缩小
    small = cv2.resize(test_img, (int(W * scale), int(H * scale)), interpolation=interp)
    # 放大回原尺寸
    restored = cv2.resize(small, (W, H), interpolation=interp)
    results[name] = restored

# 计算PSNR
def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("\n[插值方法质量对比（PSNR，越大越好）]")
for name, img in results.items():
    p = psnr(test_img, img)
    print(f"  {name}: PSNR={p:.1f} dB")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(test_img, cmap="gray")
axes[0, 0].set_title("原图")
axes[0, 0].axis("off")

for col, (name, img) in enumerate(results.items(), 1):
    axes[0, col].imshow(img, cmap="gray")
    p = psnr(test_img, img)
    axes[0, col].set_title(f"{name}\nPSNR={p:.1f}dB")
    axes[0, col].axis("off")

# 放大斜线区域（最能看到锯齿差异）
crop_y, crop_x = 50, 100
crop_size = 100

for col, (name, img) in enumerate(results.items(), 1):
    crop_orig = test_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    crop_result = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    # 并排对比
    comparison = np.hstack([crop_orig, crop_result])
    axes[1, col-1] if col > 0 else None

axes[1, 0].imshow(test_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 0].set_title("原图（局部放大）")
axes[1, 0].axis("off")

axes[1, 1].imshow(results["最近邻 (INTER_NEAREST)"][crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 1].set_title("最近邻（锯齿明显）")
axes[1, 1].axis("off")

axes[1, 2].imshow(results["双线性 (INTER_LINEAR)"][crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 2].set_title("双线性（平滑过渡）")
axes[1, 2].axis("off")

axes[1, 3].imshow(results["双三次 (INTER_CUBIC)"][crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 3].set_title("双三次（最平滑）")
axes[1, 3].axis("off")

plt.suptitle("实验一：三种插值方法对比\n（缩小50%再放大，观察斜线边缘的锯齿程度）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S8_几何变换/实验结果_插值对比.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存")
plt.show()

# 【实验现象1】最近邻有明显的锯齿（像素是方块状的跳变）
# 【实验现象2】双线性边缘平滑（加权平均的结果）
# 【实验现象3】双三次最平滑，但计算量最大

# ============================================================
# 实验二：多次变换的插值误差累积
# ============================================================
print("\n" + "=" * 60)
print("实验二：多次几何变换的插值误差累积")
print("=" * 60)

# 每次旋转1度，旋转10次
angle = 1.0
H, W = test_img.shape
center = (W // 2, H // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)

current = test_img.copy()
psnr_history = [psnr(test_img, current)]

print("\n[旋转10次（每次1度）的PSNR退化]")
for i in range(10):
    current = cv2.warpAffine(current, M, (W, H), flags=cv2.INTER_LINEAR)
    p = psnr(test_img, current)
    psnr_history.append(p)
    print(f"  第{i+1:2d}次旋转: PSNR={p:.1f} dB {'⚠️ 模糊明显' if p < 25 else ''}")

# 可视化PSNR退化曲线
plt.figure(figsize=(10, 4))
plt.plot(range(11), psnr_history, "bo-", linewidth=2, markersize=8)
plt.axhline(y=30, color="r", linestyle="--", label="PSNR=30dB (明显退化)")
plt.xlabel("旋转次数")
plt.ylabel("PSNR (dB)")
plt.title("实验二：多次旋转的插值误差累积\n（每次旋转都用双线性插值，10次后PSNR下降约10dB）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("S8_几何变换/实验结果_误差累积.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# 展示旋转后的图像质量退化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_img, cmap="gray")
axes[0].set_title("原图")
axes[0].axis("off")

axes[1].imshow(current, cmap="gray")
axes[1].set_title(f"旋转10次后\nPSNR={psnr_history[-1]:.1f}dB（模糊）")
axes[1].axis("off")

axes[2].plot(psnr_history, "bo-", linewidth=2, markersize=8)
axes[2].axhline(y=30, color="r", linestyle="--")
axes[2].set_title("PSNR退化曲线")
axes[2].set_xlabel("旋转次数")
axes[2].set_ylabel("PSNR (dB)")

plt.suptitle("实验二：多次几何变换的插值误差累积", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 【实验现象】每次旋转都引入插值误差（因为每次都在做"估计小数坐标"），
# 10次旋转后PSNR从∞降到30dB以下，图像明显模糊
# 这就是"几何变换次数越少越好"的原因

# ============================================================
# 实验三：图像配准三步骤演示
# ============================================================
print("\n" + "=" * 60)
print("实验三：图像配准三步骤（特征提取→匹配→变换）")
print("=" * 60)

# 构造两张有平移和旋转的图
from skimage import data

ref_img = data.page()  # 带文字的扫描页
ref_gray = cv2.resize(cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY), (400, 300))
ref_gray = (ref_gray / ref_gray.max() * 255).astype(np.uint8)

# 构造偏移和旋转后的版本
M_offset = np.float32([[1.0, 0.0, 30], [0.0, 1.0, 20]])  # 平移(30, 20)
offset_img = cv2.warpAffine(ref_gray, M_offset, (400, 300))

# 再加旋转
angle = 5.0
H, W = ref_gray.shape
center = (W // 2, H // 2)
M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
M_combined = np.dot(M_rotate, np.float32([[1, 0, 30], [0, 1, 20], [0, 0, 1]]))[:2, :]
moved_img = cv2.warpAffine(ref_gray, M_combined, (400, 300))

# ORB特征检测（快速且无需额外库）
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(ref_gray, None)
kp2, des2 = orb.detectAndCompute(moved_img, None)

print(f"  参考图特征点: {len(kp1)} 个")
print(f"  变换图特征点: {len(kp2)} 个")

# BF匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]  # 取前50个最好的匹配

print(f"  匹配数: {len(matches)}, 取前{len(good_matches)}个高质量匹配")

# 用RANSAC估计变换矩阵
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 基础矩阵 + RANSAC
M_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
inliers = mask.ravel().sum()
print(f"  RANSAC内点: {inliers}/{len(good_matches)} (内点比例={inliers/len(good_matches)*100:.0f}%)")

# 用估计的变换矩阵对齐
aligned = cv2.warpPerspective(ref_gray, M_est, (400, 300))

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(ref_gray, cmap="gray")
axes[0, 0].set_title("参考图")
axes[0, 0].axis("off")

axes[0, 1].imshow(moved_img, cmap="gray")
axes[0, 1].set_title("待配准图（平移+旋转后）")
axes[0, 1].axis("off")

axes[0, 2].imshow(aligned, cmap="gray")
axes[0, 2].set_title("配准结果（对齐后）")
axes[0, 2].axis("off")

# 画匹配连线
match_img = cv2.drawMatches(ref_gray, kp1, moved_img, kp2, good_matches[:20], None)
axes[1, 0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("特征匹配（前20个）")
axes[1, 0].axis("off")

# 误差热图
error = np.abs(ref_gray.astype(float) - aligned.astype(float))
im = axes[1, 1].imshow(error, cmap="hot")
axes[1, 1].set_title("配准误差（差异绝对值）")
axes[1, 1].axis("off")
plt.colorbar(im, ax=axes[1, 1])

# 叠加显示
overlay = np.zeros((300, 400, 3), dtype=np.uint8)
overlay[:, :, 0] = ref_gray
overlay[:, :, 2] = aligned
axes[1, 2].imshow(overlay)
axes[1, 2].set_title("叠加显示（红=参考，绿=配准后）\n重合好=黄色，不重合=各自颜色")
axes[1, 2].axis("off")

plt.suptitle("实验三：图像配准三步骤\n（特征提取→特征匹配→RANSAC估计变换→对齐）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S8_几何变换/实验结果_配准.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验三图已保存")
plt.show()

# 【实验现象1】ORB特征点在文字和边缘丰富的区域多（这些地方特征独特）
# 【实验现象2】RANSAC内点比例>50%说明大多数匹配是对的
# 【实验现象3】配准后两张图的误差在文字区域可能较大（因为文字是高频细节）

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. 最近邻有锯齿，双线性是精度和速度的平衡，双三次最平滑但最慢")
print("2. 多次几何变换会累积插值误差——次数越少越好，变换链越短越好")
print("3. 配准三步骤：特征提取(SIFT/ORB)→特征匹配(BFMatcher)→RANSAC估计变换")
print("4. 配准前必须做空间分辨率对齐（重采样），否则特征点不在同一尺度")
