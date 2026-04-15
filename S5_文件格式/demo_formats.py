"""
Chapter 5: Image File Formats - Demo
=====================================
Goals:
  1. Compare BMP/PNG/JPG file size and quality on same image
  2. Observe JPG lossy compression artifacts (blockiness, ringing)
  3. Verify cumulative quality loss from multiple JPG saves
  4. Understand BMP bottom-up storage Y-axis flip issue
  5. Understand format selection decision logic

Install:
  pip install opencv-python numpy matplotlib Pillow

Run:
  python demo_formats.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

_cjk_fonts = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name.lower() for k in ['noto', 'wqy', 'simsun', 'simhei', 'microsoft yahei', 'pingfang', 'heiti'])]
if _cjk_fonts:
    plt.rcParams['font.family'] = _cjk_fonts[0]
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Demo 1: Same image, different formats -> size and quality
# ============================================================
print("=" * 60)
print("Demo 1: BMP / PNG / JPG file size and quality")
print("=" * 60)

test_img = np.zeros((400, 600, 3), dtype=np.uint8)
for i in range(600):
    test_img[:, i] = [i // 3, 100 + i // 4, 255 - i // 4]

for i in range(0, 600, 8):
    test_img[:, i:i+2] = [0, 0, 0]

test_img[250:350, 100:500] = [200, 200, 200]
for i in range(100, 500, 40):
    test_img[270:330, i:i+20] = [0, 0, 0]

test_img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

os.makedirs("S5_文件格式", exist_ok=True)
bmp_path = "test.bmp"
png_path = "test.png"
jpg_high_path = "test_q95.jpg"
jpg_low_path = "test_q50.jpg"

cv2.imwrite(bmp_path, test_img_bgr)
cv2.imwrite(png_path, test_img_bgr)
cv2.imwrite(jpg_high_path, test_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(jpg_low_path, test_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])

bmp_read = cv2.imread(bmp_path)
png_read = cv2.imread(png_path)
jpg_high_read = cv2.imread(jpg_high_path)
jpg_low_read = cv2.imread(jpg_low_path)

bmp_size = os.path.getsize(bmp_path) / 1024
png_size = os.path.getsize(png_path) / 1024
jpg_high_size = os.path.getsize(jpg_high_path) / 1024
jpg_low_size = os.path.getsize(jpg_low_path) / 1024

print(f"\n[File size]")
print(f"  BMP (no compression):       {bmp_size:.1f} KB")
print(f"  PNG (lossless):            {png_size:.1f} KB  ({png_size/bmp_size*100:.1f}% of BMP)")
print(f"  JPG Q=95 (high quality):   {jpg_high_size:.1f} KB  ({jpg_high_size/bmp_size*100:.1f}% of BMP)")
print(f"  JPG Q=50 (low quality):    {jpg_low_size:.1f} KB  ({jpg_low_size/bmp_size*100:.1f}% of BMP)")

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

psnr_png = compute_psnr(test_img_bgr, png_read)
psnr_jpg_high = compute_psnr(test_img_bgr, jpg_high_read)
psnr_jpg_low = compute_psnr(test_img_bgr, jpg_low_read)

print(f"\n[Quality loss (PSNR, higher=better, inf=identical)]")
print(f"  PNG vs original:    {psnr_png:.1f} dB  {'[Lossless]' if psnr_png > 50 else ''}")
print(f"  JPG Q=95 vs orig:   {psnr_jpg_high:.1f} dB")
print(f"  JPG Q=50 vs orig:   {psnr_jpg_low:.1f} dB")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

images = [
    (test_img_bgr, f"Original ({bmp_size:.0f}KB)"),
    (png_read, f"PNG ({png_size:.0f}KB, PSNR={psnr_png:.0f})"),
    (jpg_high_read, f"JPG Q=95 ({jpg_high_size:.0f}KB, PSNR={psnr_jpg_high:.0f})"),
    (jpg_low_read, f"JPG Q=50 ({jpg_low_size:.0f}KB, PSNR={psnr_jpg_low:.0f})"),
]

for col, (img, title) in enumerate(images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, col].imshow(img_rgb)
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")
    crop = img_rgb[250:350, 100:500]
    axes[1, col].imshow(crop)
    axes[1, col].set_title("Text region zoom", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("Demo 1: File Format Comparison - Size vs Quality\n(PNG=lossless, JPG Q=50=obvious block artifacts on text)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_formats.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_formats.png")
plt.close()

# ============================================================
# Demo 2: Cumulative JPG quality loss from multiple saves
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Cumulative JPG quality loss (re-save degradation)")
print("=" * 60)

iterations = [0, 1, 2, 5, 10, 20]
psnrs = []
current_img = jpg_high_read.copy()

for i in iterations:
    if i == 0:
        psnrs.append(compute_psnr(test_img_bgr, current_img))
        print(f"  Iteration {i:2d}: PSNR={psnrs[-1]:.1f} dB (baseline)")
    else:
        current_img = current_img.copy()
        temp_path = f"iter_{i}.jpg"
        cv2.imwrite(temp_path, current_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        current_img = cv2.imread(temp_path)
        psnr = compute_psnr(test_img_bgr, current_img)
        psnrs.append(psnr)
        flag = " [Degraded]" if psnr < 25 else ""
        print(f"  Iteration {i:2d}: PSNR={psnr:.1f} dB{flag}")
        os.remove(temp_path)

plt.figure(figsize=(10, 4))
plt.plot(iterations, psnrs, "bo-", linewidth=2, markersize=8)
plt.axhline(y=30, color="r", linestyle="--", label="PSNR=30dB (visible degradation)")
plt.axhline(y=40, color="orange", linestyle="--", label="PSNR=40dB (mild degradation)")
plt.xlabel("Save iterations (Q=85 each)")
plt.ylabel("PSNR (dB)")
plt.title("Demo 2: Cumulative JPG Quality Loss\n(Every re-save degrades quality - never use JPG for intermediate data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("demo_jpg_loss.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_jpg_loss.png")
plt.close()

# ============================================================
# Demo 3: BMP bottom-up storage (Y-axis flip concept)
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: BMP bottom-up storage (Y-axis flip)")
print("=" * 60)

flip_test = np.zeros((200, 400, 3), dtype=np.uint8)
flip_test[:100, :] = [255, 0, 0]
flip_test[100:, :] = [0, 0, 255]
flip_test_bgr = cv2.cvtColor(flip_test, cv2.COLOR_RGB2BGR)
flip_bmp_path = "flip_test.bmp"
cv2.imwrite(flip_bmp_path, flip_test_bgr)

# Read with PIL (correct handling)
pil_img = np.array(Image.open(flip_bmp_path))

# Read raw bytes to demonstrate storage order
with open(flip_bmp_path, "rb") as f:
    f.seek(54)
    raw_data = f.read()

row_size = (400 * 3 + 3) // 4 * 4
wrong_order = np.frombuffer(raw_data[:200 * row_size], dtype=np.uint8)
wrong_order = wrong_order.reshape(200, row_size)[:, :1200].reshape(200, 400, 3)
wrong_order = wrong_order[:, :, ::-1]
correct_order = wrong_order[::-1, :, :]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(flip_test)
axes[0].set_title("Original\n(Red=top, Blue=bottom)")
axes[0].axis("off")

axes[1].imshow(wrong_order)
axes[1].set_title("Wrong interpretation of BMP order\n(Blue=top, Red=bottom - flipped!)")
axes[1].axis("off")

axes[2].imshow(correct_order)
axes[2].set_title("Correct interpretation\n(Red=top, Blue=bottom)")
axes[2].axis("off")

plt.suptitle("Demo 3: BMP Bottom-Up Storage - Y-Axis Flip\n(BMP stores rows from bottom to top - writing a raw parser without knowing this flips the image)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_bmp_flip.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_bmp_flip.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. PNG = best lossless choice (smaller than BMP, identical quality)")
print("2. JPG Q=50 shows obvious block artifacts and ringing on text/edges")
print("3. Multiple JPG saves accumulate quality loss -> never use JPG for intermediate data")
print("4. BMP stores rows bottom-to-top -> writing raw parsers needs this knowledge")
