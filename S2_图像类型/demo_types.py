"""
Chapter 2: Image Types - Demo
=============================
Goals:
  1. Understand binary/grayscale/RGB/indexed image differences in data
  2. Observe indexed image palette trap
  3. Understand RGB-to-grayscale weighted formula

Install:
  pip install opencv-python numpy matplotlib

Run:
  python demo_types.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

import cv2
import numpy as np
import matplotlib.pyplot as plt

_cjk_fonts = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name.lower() for k in ['noto', 'wqy', 'simsun', 'simhei', 'microsoft yahei', 'pingfang', 'heiti'])]
if _cjk_fonts:
    plt.rcParams['font.family'] = _cjk_fonts[0]
    print(f"[Font] Using CJK font: {_cjk_fonts[0]}")
else:
    print("[Font] No CJK font found, labels use English fallback")
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Demo 1: Four image types - numeric structure
# ============================================================
print("=" * 60)
print("Demo 1: Binary / Grayscale / RGB / Indexed numeric structure")
print("=" * 60)

canvas = np.zeros((200, 400), dtype=np.uint8)
canvas[:, 200:] = 255

binary = (canvas > 128).astype(np.uint8) * 255
unique_vals = np.unique(binary)
print(f"[Binary] Unique values: {unique_vals}")

gray = canvas.copy()
unique_vals_gray = np.unique(gray)
print(f"[Grayscale] Unique count: {len(unique_vals_gray)}")
print(f"[Grayscale] Range: {gray.min()} ~ {gray.max()}")

gradient = np.linspace(0, 255, 400, dtype=np.uint8)
gradient_img = np.tile(gradient, (100, 1))
unique_gradient = np.unique(gradient_img)
print(f"[Gradient] Unique count: {len(unique_gradient)}")

rgb_canvas = np.zeros((200, 400, 3), dtype=np.uint8)
rgb_canvas[:, :200] = [30, 30, 30]
rgb_canvas[:, 200:] = [180, 50, 50]
print(f"\n[RGB] Shape: {rgb_canvas.shape} (H x W x C)")
print(f"[RGB] Left pixel: {rgb_canvas[0, 0]}")
print(f"[RGB] Right pixel: {rgb_canvas[0, 201]}")

brightened = np.clip(rgb_canvas.astype(int) + 50, 0, 255).astype(np.uint8)
ratio_r = brightened[0, 201, 0] / rgb_canvas[0, 201, 0]
ratio_g = brightened[0, 201, 1] / rgb_canvas[0, 201, 1]
ratio_b = brightened[0, 201, 2] / rgb_canvas[0, 201, 2]
print(f"[RGB+50] Channel ratios: R={ratio_r:.2f}, G={ratio_g:.2f}, B={ratio_b:.2f}")
print("[Insight] All three channels scale proportionally -> hue preserved under illumination change")

# ============================================================
# Demo 2: RGB-to-grayscale formula differences
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: RGB-to-grayscale formula (OpenCV vs PIL vs MATLAB)")
print("=" * 60)

test_rgb = np.array([[[200, 100, 50]]], dtype=np.uint8)
opencv_gray = int(0.114 * test_rgb[0, 0, 0] + 0.587 * test_rgb[0, 0, 1] + 0.299 * test_rgb[0, 0, 2])
pil_gray = int(0.299 * test_rgb[0, 0, 2] + 0.587 * test_rgb[0, 0, 1] + 0.114 * test_rgb[0, 0, 0])

print(f"[Pixel] B={test_rgb[0,0,0]}, G={test_rgb[0,0,1]}, R={test_rgb[0,0,2]}")
print(f"[OpenCV] Gray = 0.114*B + 0.587*G + 0.299*R = {opencv_gray}")
print(f"[PIL]    Gray = 0.299*R + 0.587*G + 0.114*B = {pil_gray}")
print(f"[Diff] OpenCV vs PIL differ by: {abs(opencv_gray - pil_gray)}")
print("[Warning] Different libraries use different weights -> pixel-level comparison across libs is risky")

# ============================================================
# Demo 3: Indexed image palette trap
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Indexed image palette trap")
print("=" * 60)

fake_indexed = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)

palette = np.zeros((256, 3), dtype=np.uint8)
palette[1] = [255, 0, 0]
palette[2] = [0, 255, 0]
palette[3] = [0, 0, 255]
palette[4] = [255, 255, 0]
palette[5] = [255, 0, 255]
palette[6] = [0, 255, 255]
palette[7] = [255, 255, 255]
palette[8] = [0, 0, 0]
palette[9] = [128, 128, 128]

wrong_gray = fake_indexed.astype(np.uint8)
print(f"[Wrong] Treating index as gray value: {wrong_gray.tolist()}")
print(f"  -> Index 1 becomes gray=1 (near black), but it should be red")

correct_rgb = palette[fake_indexed]
print(f"[Correct] Palette lookup:")
print(f"  Index 1 -> {correct_rgb[0,0]} (red)")
print(f"  Index 2 -> {correct_rgb[0,1]} (green)")
print(f"  Index 3 -> {correct_rgb[0,2]} (blue)")
print("[Trap] Always apply palette before rendering indexed images!")

# ============================================================
# Demo 4: Visual comparison of four image types
# ============================================================
print("\n" + "=" * 60)
print("Demo 4: Visual comparison (four types)")
print("=" * 60)

test_img = np.zeros((300, 600, 3), dtype=np.uint8)
test_img[:100, :200, 2] = np.linspace(0, 255, 200, dtype=np.uint8)[np.newaxis, :]
test_img[:100, :200, 0] = 200
test_img[:100, 200:, 1] = 200
test_img[100:, :200, 2] = 200
noise_patch = np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8)
test_img[100:, 200:] = noise_patch

gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
_, binary_test = cv2.threshold(gray_test, 130, 255, cv2.THRESH_BINARY)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("RGB Color Image (3 channels)")
axes[0].axis("off")

axes[1].imshow(gray_test, cmap="gray")
axes[1].set_title("Grayscale Image (1 channel)")
axes[1].axis("off")

axes[2].imshow(binary_test, cmap="gray")
axes[2].set_title("Binary Image (0 or 255 only)")
axes[2].axis("off")

axes[3].hist(gray_test.ravel(), bins=50, color="gray")
axes[3].set_title("Grayscale Histogram (for threshold selection)")
axes[3].set_xlabel("Pixel value")
axes[3].set_ylabel("Count")

plt.suptitle("Demo 4: Four Image Types - Numeric Structure", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("S2_图像类型/demo_output.png", dpi=150, bbox_inches="tight")
print("[Saved] -> S2_图像类型/demo_output.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Binary: 0/255 only; Grayscale: 0-255; RGB: 3 channels")
print("2. RGB channels are coupled -> illumination change scales all 3 proportionally")
print("3. Grayscale formula differs by library (OpenCV vs PIL: diff by 10+)")
print("4. Indexed images need palette lookup, else index=gray=wrong color")
