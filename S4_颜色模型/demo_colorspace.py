"""
Chapter 4: Color Models - Demo
================================
Goals:
  1. Observe RGB channel coupling under illumination changes
  2. Verify HSV H channel is illumination-invariant (why HSV for color segmentation)
  3. Understand YUV luminance-chrominance separation
  4. Observe that low saturation makes H unreliable

Install:
  pip install opencv-python numpy matplotlib

Run:
  python demo_colorspace.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

_cjk_fonts = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name.lower() for k in ['noto', 'wqy', 'simsun', 'simhei', 'microsoft yahei', 'pingfang', 'heiti'])]
if _cjk_fonts:
    plt.rcParams['font.family'] = _cjk_fonts[0]
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Demo 1: RGB channel coupling under illumination
# ============================================================
print("=" * 60)
print("Demo 1: RGB channel coupling under illumination change")
print("=" * 60)

rgb_orig = np.zeros((200, 400, 3), dtype=np.uint8)
rgb_orig[:, :200] = [180, 60, 40]   # warm orange-ish
rgb_orig[:, 200:] = [60, 120, 200]  # cool blue-ish

print(f"[Original] Left: B={rgb_orig[0,0,0]} G={rgb_orig[0,0,1]} R={rgb_orig[0,0,2]}")
print(f"[Original] Right: B={rgb_orig[0,201,0]} G={rgb_orig[0,201,1]} R={rgb_orig[0,201,2]}")

rgb_bright = np.clip(rgb_orig.astype(int) + 50, 0, 255).astype(np.uint8)
print(f"\n[Brightness+50] Left: B={rgb_bright[0,0,0]} G={rgb_bright[0,0,1]} R={rgb_bright[0,0,2]}")
print(f"[Brightness+50] Right: B={rgb_bright[0,201,0]} G={rgb_bright[0,201,1]} R={rgb_bright[0,201,2]}")

for side, idx in [("Left", 0), ("Right", 201)]:
    r_orig = rgb_orig[0, idx, 2] / (rgb_orig[0, idx].sum() + 1e-6)
    r_bright = rgb_bright[0, idx, 2] / (rgb_bright[0, idx].sum() + 1e-6)
    print(f"[{side}] R-channel ratio: orig={r_orig:.3f}, bright={r_bright:.3f} -> unchanged, hue preserved")

# ============================================================
# Demo 2: HSV H channel is illumination-invariant
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: HSV H channel is illumination-invariant")
print("=" * 60)

hsv_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2HSV)
hsv_bright = cv2.cvtColor(rgb_bright, cv2.COLOR_BGR2HSV)

print(f"[HSV Original] Left  H={hsv_orig[0,0,0]:.1f} S={hsv_orig[0,0,1]} V={hsv_orig[0,0,2]}")
print(f"[HSV Bright]   Left  H={hsv_bright[0,0,0]:.1f} S={hsv_bright[0,0,1]} V={hsv_bright[0,0,2]}")
print(f"[HSV Original] Right H={hsv_orig[0,201,0]:.1f} S={hsv_orig[0,201,1]} V={hsv_orig[0,201,2]}")
print(f"[HSV Bright]   Right H={hsv_bright[0,201,0]:.1f} S={hsv_bright[0,201,1]} V={hsv_bright[0,201,2]}")
print("\n[Insight] H (Hue) stays nearly same -> HSV H is illumination-invariant")
print("[Insight] V (Value) changes a lot -> V = brightness")

# Create test image with known colors
test_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
test_rgb[:100, :133, 2] = 255    # Red region
test_rgb[:100, 133:266, 1] = 255  # Green region
test_rgb[:100, 266:, 0] = 255    # Blue region
test_rgb[100:, :, :] = [128, 128, 128]  # Gray (low saturation)

hsv_test = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2HSV)
gray_test = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2GRAY)

# ============================================================
# Demo 3: YUV luminance-chrominance separation
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: YUV luminance-chrominance separation")
print("=" * 60)

yuv = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2YUV)
print(f"[YUV] Y(luma) range: {yuv[:,:,0].min()}-{yuv[:,:,0].max()} -> brightness")
print(f"[YUV] U range: {yuv[:,:,1].min()}-{yuv[:,:,1].max()} -> blue-difference")
print(f"[YUV] V range: {yuv[:,:,2].min()}-{yuv[:,:,2].max()} -> red-difference")
print("[Insight] Y carries brightness only -> drop U/V for B&W, compress U/V for video")

# ============================================================
# Demo 4: Low saturation -> H becomes unreliable
# ============================================================
print("\n" + "=" * 60)
print("Demo 4: Low saturation -> H value is meaningless")
print("=" * 60)

gray_bgr = cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR)
gray_hsv_converted = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2HSV)
print(f"[Gray->HSV] S={gray_hsv_converted[0,0,1]} (S=0 means no saturation -> H is meaningless!)")
print("[Warning] When S is near 0, H value has no physical meaning")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 1: RGB illumination change
rgb_display = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2RGB)
axes[0, 0].imshow(rgb_display)
axes[0, 0].set_title("RGB Original (colors: R/G/B/Gray)")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(rgb_bright, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("RGB + Brightness 50\n(hue preserved, all channels scaled)")
axes[0, 1].axis("off")

for c, name in [(2, "R"), (1, "G"), (0, "B")]:
    ch = rgb_orig[:, :, c]
    axes[0, 2].plot(range(400), ch[100, :], label=name, linewidth=2)
axes[0, 2].set_title("RGB channel values (row 100)")
axes[0, 2].legend()
axes[0, 2].set_xlabel("Column")

axes[0, 3].imshow(yuv[:,:,0], cmap="gray")
axes[0, 3].set_title("YUV: Y (Luminance) - only brightness")
axes[0, 3].axis("off")

# Row 2: HSV decomposition
axes[1, 0].imshow(cv2.cvtColor(hsv_test.astype(np.uint8), cv2.COLOR_HSV2BGR))
axes[1, 0].set_title("HSV Original (reconstructed)")
axes[1, 0].axis("off")

axes[1, 1].imshow(hsv_test[:,:,0], cmap="hsv")
axes[1, 1].set_title("H (Hue) - color type, illumination-invariant")
axes[1, 1].axis("off")

axes[1, 2].imshow(hsv_test[:,:,1], cmap="gray")
axes[1, 2].set_title("S (Saturation) - color purity, 0=gray")
axes[1, 2].axis("off")

axes[1, 3].imshow(hsv_test[:,:,2], cmap="gray")
axes[1, 3].set_title("V (Value) - brightness")
axes[1, 3].axis("off")

# Row 3: Color segmentation demo
hsv_test_u8 = hsv_test.astype(np.uint8)
mask_red = cv2.inRange(hsv_test_u8, (0, 100, 100), (10, 255, 255))
mask_green = cv2.inRange(hsv_test_u8, (40, 50, 50), (80, 255, 255))
mask_blue = cv2.inRange(hsv_test_u8, (100, 50, 50), (140, 255, 255))
mask_all = mask_red | mask_green | mask_blue
result = cv2.bitwise_and(rgb_display, rgb_display, mask=mask_all)

axes[2, 0].imshow(rgb_display)
axes[2, 0].set_title("Original")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Color segmentation via HSV mask\n(H-based -> illumination invariant)")
axes[2, 1].axis("off")

axes[2, 2].imshow(gray_test, cmap="gray")
axes[2, 2].set_title("Grayscale (no hue -> hard to segment colors)")
axes[2, 2].axis("off")

axes[2, 3].hist(rgb_display[:,:,0].ravel(), bins=50, alpha=0.5, color='b', label='B')
axes[2, 3].hist(rgb_display[:,:,1].ravel(), bins=50, alpha=0.5, color='g', label='G')
axes[2, 3].hist(rgb_display[:,:,2].ravel(), bins=50, alpha=0.5, color='r', label='R')
axes[2, 3].set_title("RGB channel histograms")
axes[2, 3].legend()

plt.suptitle("Demo: Color Models - RGB/HSV/YUV Decomposition", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("S4_颜色模型/demo_output.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> S4_颜色模型/demo_output.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. RGB channels are coupled -> illumination changes all 3 proportionally")
print("2. HSV H is illumination-invariant -> best for color segmentation")
print("3. YUV separates luminance (Y) from chrominance (U/V) -> video compression uses this")
print("4. H is meaningless when S=0 (grayscale) -> always check S before using H")
