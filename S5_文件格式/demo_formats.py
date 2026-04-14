"""
第五章：图像文件格式基础 - 实验演示
==================================
实验目标：
  1. 对比BMP/PNG/JPG在同一张图上的文件大小和质量差异
  2. 观察JPG有损压缩的块效应和振铃效应
  3. 验证多次JPG保存的累积质量损失
  4. 理解BMP从下到上存储的Y轴翻转问题
  5. 理解格式选择的决策逻辑

实验准备：
  pip install opencv-python numpy matplotlib Pillow

运行：
  python demo_formats.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 确保输出目录存在
os.makedirs("S5_文件格式", exist_ok=True)

# ============================================================
# 实验一：同一张图，不同格式的文件大小和质量对比
# ============================================================
print("=" * 60)
print("实验一：同一张图，BMP / PNG / JPG 文件大小和质量对比")
print("=" * 60)

# 创建一张测试图：包含渐变、纹理、文字边缘
test_img = np.zeros((400, 600, 3), dtype=np.uint8)

# 背景：渐变色
for i in range(600):
    test_img[:, i] = [i // 3, 100 + i // 4, 255 - i // 4]

# 添加高频纹理（模拟细节）
for i in range(0, 600, 8):
    test_img[:, i:i+2] = [0, 0, 0]  # 黑色细线（文字边缘类）

# 添加文字区域（文字最容易暴露JPG的振铃效应）
test_img[250:350, 100:500] = [200, 200, 200]  # 灰色背景
# 在灰色背景上画"假文字"（黑色块）
for i in range(100, 500, 40):
    test_img[270:330, i:i+20] = [0, 0, 0]

# 保存为三种格式
test_img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

bmp_path = "S5_文件格式/test_format.bmp"
png_path = "S5_文件格式/test_format.png"
jpg_path_high = "S5_文件格式/test_format_q95.jpg"
jpg_path_low = "S5_文件格式/test_format_q50.jpg"

cv2.imwrite(bmp_path, test_img_bgr)
cv2.imwrite(png_path, test_img_bgr)
cv2.imwrite(jpg_path_high, test_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(jpg_path_low, test_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])

# 读取回来
bmp_read = cv2.imread(bmp_path)
png_read = cv2.imread(png_path)
jpg_high_read = cv2.imread(jpg_path_high)
jpg_low_read = cv2.imread(jpg_path_low)

# 计算文件大小
bmp_size = os.path.getsize(bmp_path) / 1024
png_size = os.path.getsize(png_path) / 1024
jpg_high_size = os.path.getsize(jpg_path_high) / 1024
jpg_low_size = os.path.getsize(jpg_path_low) / 1024

print(f"\n[文件大小对比]")
print(f"  BMP (无压缩):       {bmp_size:.1f} KB")
print(f"  PNG (无损压缩):     {png_size:.1f} KB  (占BMP的{png_size/bmp_size*100:.1f}%)")
print(f"  JPG Q=95 (高质):    {jpg_high_size:.1f} KB  (占BMP的{jpg_high_size/bmp_size*100:.1f}%)")
print(f"  JPG Q=50 (低质):    {jpg_low_size:.1f} KB  (占BMP的{jpg_low_size/bmp_size*100:.1f}%)")

# 计算PSNR（峰值信噪比）——衡量图像质量损失
def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_png = compute_psnr(test_img_bgr, png_read)
psnr_jpg_high = compute_psnr(test_img_bgr, jpg_high_read)
psnr_jpg_low = compute_psnr(test_img_bgr, jpg_low_read)

print(f"\n[质量损失对比 (PSNR，越大越好，∞=完全相同)]")
print(f"  PNG vs 原图:        {psnr_png:.1f} dB  {'✓ 无损' if psnr_png > 50 else ''}")
print(f"  JPG Q=95 vs 原图:   {psnr_jpg_high:.1f} dB")
print(f"  JPG Q=50 vs 原图:   {psnr_jpg_low:.1f} dB")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

images = [
    (test_img_bgr, "原图"),
    (png_read, f"PNG ({png_size:.0f}KB, PSNR={psnr_png:.0f})"),
    (jpg_high_read, f"JPG Q=95 ({jpg_high_size:.0f}KB, PSNR={psnr_jpg_high:.0f})"),
    (jpg_low_read, f"JPG Q=50 ({jpg_low_size:.0f}KB, PSNR={psnr_jpg_low:.0f})"),
]

for col, (img, title) in enumerate(images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, col].imshow(img_rgb)
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

    # 下排：放大文字区域（最容易看到块效应和振铃效应）
    crop = img_rgb[250:350, 100:500]
    axes[1, col].imshow(crop)
    axes[1, col].set_title(f"文字区域放大", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("实验一：文件格式对比\n（大小、质量、文字区域块效应）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S5_文件格式/实验结果_格式对比.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验一图已保存")
plt.show()

# 【实验现象1】PNG无损，PSNR=∞，文件比BMP小很多但质量相同
# 【实验现象2】JPG Q=95有轻微损失，PSNR约40dB，肉眼难辨
# 【实验现象3】JPG Q=50有明显块效应，尤其在文字边缘

# ============================================================
# 实验二：JPG多次保存的累积质量损失
# ============================================================
print("\n" + "=" * 60)
print("实验二：JPG多次保存的累积质量损失")
print("=" * 60)

# 从高质量JPG开始，反复保存低质量版本
iterations = [0, 1, 2, 5, 10, 20]
psnrs = []

current_img = jpg_high_read.copy()  # 从Q=95开始

for i in iterations:
    if i == 0:
        psnrs.append(compute_psnr(test_img_bgr, current_img))
        print(f"  第{i:2d}次保存: PSNR={psnrs[-1]:.1f} dB")
    else:
        # 用Q=85保存（模拟每次处理都重新编码）
        current_img = current_img.copy()
        temp_path = f"S5_文件格式/iter_{i}.jpg"
        cv2.imwrite(temp_path, current_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        current_img = cv2.imread(temp_path)
        psnr = compute_psnr(test_img_bgr, current_img)
        psnrs.append(psnr)
        print(f"  第{i:2d}次保存: PSNR={psnr:.1f} dB {'⚠️ 严重退化' if psnr < 25 else ''}")
        os.remove(temp_path)

print(f"\n[累积损失总结]")
print(f"  0次: {psnrs[0]:.1f} dB (基准)")
print(f"  5次: {psnrs[4]:.1f} dB (下降{psnrs[0]-psnrs[4]:.1f} dB)")
print(f"  20次: {psnrs[5]:.1f} dB (下降{psnrs[0]-psnrs[5]:.1f} dB)")
# 【实验现象】每次保存都累积损失，20次后PSNR从40dB跌到25dB左右
# 这就是"机器视觉中间数据不能用JPG"的根本原因

# 绘制PSNR退化曲线
plt.figure(figsize=(10, 4))
plt.plot(iterations, psnrs, "bo-", linewidth=2, markersize=8)
plt.axhline(y=30, color="r", linestyle="--", label="PSNR=30dB (明显退化阈值)")
plt.axhline(y=40, color="orange", linestyle="--", label="PSNR=40dB (轻微退化)")
plt.xlabel("保存次数")
plt.ylabel("PSNR (dB)")
plt.title("实验二：JPG多次保存的累积质量损失\n（每次用Q=85重新编码）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("S5_文件格式/实验结果_JPG累积损失.png", dpi=150, bbox_inches="tight")
print("\n[保存] 实验二图已保存")
plt.show()

# ============================================================
# 实验三：BMP的Y轴翻转问题
# ============================================================
print("\n" + "=" * 60)
print("实验三：BMP存储顺序（从下到上）导致的Y轴翻转")
print("=" * 60)

# 构造一张有明显方向性的图：上面是红，下面是蓝
flip_test = np.zeros((200, 400, 3), dtype=np.uint8)
flip_test[:100, :] = [255, 0, 0]    # 上半：红色
flip_test[100:, :] = [0, 0, 255]    # 下半：蓝色

# 保存BMP
flip_test_bgr = cv2.cvtColor(flip_test, cv2.COLOR_RGB2BGR)
flip_bmp_path = "S5_文件格式/flip_test.bmp"
cv2.imwrite(flip_bmp_path, flip_test_bgr)

# 用PIL读取（正确处理了BMP的存储顺序）
pil_img = np.array(Image.open(flip_bmp_path))

# 用OpenCV直接读取数组方式模拟"不知道BMP从下到上"的情况
# 实际上cv2.imread已经处理了这个问题，这里演示存储顺序的概念
# 手动写一个raw BMP读取，模拟错误理解存储顺序的情况

# 用numpy直接读取原始字节（不解析BMP头），模拟错误的"从上到下"理解
# 读取BMP的像素数据区（跳过54字节头）
with open(flip_bmp_path, "rb") as f:
    f.seek(54)  # 跳过BMP文件头+信息头
    raw_data = f.read()

# 解析像素数据：BMP从下到上存储，每行4字节对齐
# 正确理解：从最后一行开始读
# 错误理解（如果以为从上到下）：从第一行开始读

row_size = (400 * 3 + 3) // 4 * 4  # 4字节对齐
wrong_order = np.frombuffer(raw_data[:200 * row_size], dtype=np.uint8)
wrong_order = wrong_order.reshape(200, row_size)[:, :1200].reshape(200, 400, 3)
wrong_order = wrong_order[:, :, ::-1]  # BGR→RGB

# 正确顺序：从下到上读
correct_order = wrong_order[::-1, :, :]  # 上下翻转

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(flip_test)
axes[0].set_title("原始图像\n(上红下蓝)")
axes[0].axis("off")

axes[1].imshow(wrong_order)
axes[1].set_title("错误理解BMP存储顺序\n(上蓝下红，翻转了)")
axes[1].axis("off")

axes[2].imshow(correct_order)
axes[2].set_title("正确理解BMP存储顺序\n(上红下蓝)")
axes[2].axis("off")

plt.suptitle("实验三：BMP从下到上存储导致的Y轴翻转问题\n（如果你自己写BMP解析器容易踩这个坑）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("S5_文件格式/实验结果_BMP翻转.png", dpi=150, bbox_inches="tight")
print("[保存] 实验三图已保存")
plt.show()

print("\n" + "=" * 60)
print("本章实验结论汇总：")
print("=" * 60)
print("1. PNG是无损压缩的最佳选择——文件比BMP小，质量相同")
print("2. JPG有损，Q=50有明显的块效应和振铃效应")
print("3. 多次JPG保存累积损失——中间数据绝对不要用JPG")
print("4. BMP从下到上存储——自己写解析