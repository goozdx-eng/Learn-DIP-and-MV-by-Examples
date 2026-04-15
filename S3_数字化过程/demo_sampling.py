"""
Chapter 3: Image Digitization - Demo
=====================================
Goals:
  1. Observe sampling density effect on image detail
  2. Observe quantization levels on gradient smoothness
  3. Understand moire pattern from undersampling
  4. Confirm sampling and quantization are independent

Install:
  pip install opencv-python numpy matplotlib scikit-image

Run:
  python demo_sampling.py
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
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Demo 1: Sampling density -> spatial resolution
# ============================================================
print("=" * 60)
print("Demo 1: Sampling density (pixel count) -> image quality")
print("=" * 60)

# Use synthetic image (no external dependency)
original_gray = np.zeros((256, 512), dtype=np.uint8)
# Horizontal gradient
original_gray[:] = np.linspace(0, 255, 512, dtype=np.uint8)[np.newaxis, :]
# Add some structure (text-like edges)
original_gray[80:100, :] = 200
original_gray[150:170, 100:400] = 80
original_gray[100:150, 200] = 255
original_gray[100:150, 300] = 0
H, W = original_gray.shape
print(f"[Original] Size: {H}x{W} = {H*W:,} pixels (synthetic test image)")

sampling_rates = [1.0, 0.5, 0.25, 0.125, 0.0625]
sampled_images = []
labels = []

for rate in sampling_rates:
    new_h = int(H * rate)
    new_w = int(W * rate)
    small = cv2.resize(original_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    sampled_images.append(restored)
    labels.append(f"Rate={rate:.2%} ({new_w}x{new_h})")
    print(f"  {labels[-1]} -> pixels: {new_h*new_w:,} ({rate:.2%} of original)")

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (img, label) in enumerate(zip(sampled_images, labels)):
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(label, fontsize=9)
    axes[i].axis("off")

plt.suptitle("Demo 1: Sampling Density -> Spatial Resolution\n(Same image, different pixel counts)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_output.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_output.png")
plt.close()

# ============================================================
# Demo 2: Quantization levels -> gradient smoothness
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Quantization levels (gray levels) -> gradient smoothness")
print("=" * 60)

gradient = np.linspace(0, 255, 512, dtype=np.uint8)
gradient_img = np.tile(gradient, (100, 1))

quantization_levels = [256, 64, 16, 4, 2]
quantized_images = []

for levels in quantization_levels:
    step = 256 // levels
    quantized = (gradient_img // step) * step
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    quantized_images.append(quantized)
    actual_levels = len(np.unique(quantized))
    print(f"  Level={levels:3d} -> actual unique: {actual_levels:3d}")

fig, axes = plt.subplots(2, 5, figsize=(20, 6))
for i, (img, levels) in enumerate(zip(quantized_images, quantization_levels)):
    axes[0, i].imshow(img, cmap="gray", aspect="auto")
    axes[0, i].set_title(f"Levels={levels}", fontsize=10)
    axes[0, i].axis("off")
    center_slice = img[:, 200:312]
    axes[1, i].imshow(center_slice, cmap="gray", aspect="auto")
    axes[1, i].set_title("Zoomed center (banding visible)", fontsize=9)
    axes[1, i].axis("off")

plt.suptitle("Demo 2: Quantization Levels -> Gradient Smoothness\n(256 levels=smooth, 2 levels=binary)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_quant.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_quant.png")
plt.close()

# ============================================================
# Demo 3: Moire pattern (aliasing from undersampling)
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Moire pattern (undersampling aliasing)")
print("=" * 60)

stripe_freq = 40
stripes = np.zeros((300, 600), dtype=np.uint8)
x = np.arange(600)
stripes = ((x % stripe_freq) < (stripe_freq // 2)).astype(np.uint8) * 255
stripes = np.tile(stripes, (300, 1))

print(f"[Pattern] Stripe period: {stripe_freq} pixels")
print(f"[Theory] Nyquist: sampling freq must be > 2x signal freq")

camera_resolutions = [600, 150, 75, 37, 19]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, res in enumerate(camera_resolutions):
    small = cv2.resize(stripes, (res, 300), interpolation=cv2.INTER_AREA)
    captured = cv2.resize(small, (600, 300), interpolation=cv2.INTER_NEAREST)
    axes[i].imshow(captured, cmap="gray")
    ratio = 600 / res
    axes[i].set_title(f"1/{ratio:.0f} resolution\n({ratio:.1f} orig px/pixel)", fontsize=9)
    axes[i].axis("off")

plt.suptitle("Demo 3: Undersampling -> Moire Pattern\n(High-freq stripes folded to low-freq waves)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_moire.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_moire.png")
plt.close()

# ============================================================
# Demo 4: Sampling and quantization are independent
# ============================================================
print("\n" + "=" * 60)
print("Demo 4: Sampling and quantization are independent")
print("=" * 60)

# Use synthetic image instead of data.camera() (no skimage dependency)
test_img = np.zeros((256, 256), dtype=np.uint8)
test_img[:] = np.linspace(0, 255, 256, dtype=np.uint8)[np.newaxis, :]
test_img[80:120, :] = 180
test_img[140:180, :] = 50




fig, axes = plt.subplots(4, 4, figsize=(16, 16))

sample_rates = [1.0, 0.5, 0.25, 0.125]
quant_levels = [256, 64, 16, 4]

for row, sr in enumerate(sample_rates):
    for col, ql in enumerate(quant_levels):
        h, w = test_img.shape
        new_h, new_w = int(h * sr), int(w * sr)
        sampled = cv2.resize(test_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if ql < 256:
            step = 256 // ql
            quantized = (sampled // step) * step
            processed = np.clip(quantized, 0, 255).astype(np.uint8)
        else:
            processed = sampled
        axes[row, col].imshow(processed, cmap="gray")
        axes[row, col].set_title(f"Samp={sr:.0%} x Quant={ql}", fontsize=8)
        axes[row, col].axis("off")

plt.suptitle("Demo 4: Sampling vs Quantization - Independent Effects\n(Each cell = unique combination of sampling rate and quantization level)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_independent.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_independent.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Sampling rate down -> spatial detail loss -> blur/moire")
print("2. Quantization levels down -> fewer gray steps -> banding")
print("3. Sampling and quantization are independent, each controls one quality dimension")
print("4. Moire = undersampling -> high freq folded to low freq (Nyquist theorem)")
