"""
Chapter 6: Image Operation Types - Demo
========================================
Goals:
  1. Verify point operations (inversion, threshold, stretch) are "pixel-only" (no neighbors)
  2. Compare mean filter vs median filter on salt-and-pepper noise
  3. Verify superposition principle: linear=YES, nonlinear=NO
  4. Understand LUT (lookup table) acceleration

Install:
  pip install opencv-python numpy matplotlib

Run:
  python demo_operation_types.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

_cjk_fonts = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name.lower() for k in ['noto', 'wqy', 'simsun', 'simhei', 'microsoft yahei', 'pingfang', 'heiti'])]
if _cjk_fonts:
    plt.rcParams['font.family'] = _cjk_fonts[0]
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("S6_运算类型", exist_ok=True)

# ============================================================
# Demo 1: Point operations - pixel-only, no neighbors
# ============================================================
print("=" * 60)
print("Demo 1: Point operations (inversion/stretch/gamma)")
print("=" * 60)

test_img = np.zeros((200, 400), dtype=np.uint8)
test_img[:, :200] = np.linspace(0, 255, 200, dtype=np.uint8)[np.newaxis, :]
test_img[:, 200:] = 128

inv_img = 255 - test_img

low, high = 50, 200
stretched = np.clip((test_img.astype(float) - low) / (high - low) * 255, 0, 255).astype(np.uint8)

gamma = 2.0
gamma_corrected = np.power(test_img.astype(float) / 255.0, 1/gamma) * 255
gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
operations = [
    (test_img, "Original"),
    (inv_img, "Inversion\n(255-value)"),
    (stretched, "Contrast Stretch\n[50,200]->[0,255]"),
    (gamma_corrected, "Gamma Correct\n(gamma=2.0 brightens dark)"),
]

for col, (img, title) in enumerate(operations):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")
    profile = img[100, :]
    axes[1, col].plot(profile, color="gray", linewidth=2)
    axes[1, col].set_ylim(0, 255)
    axes[1, col].set_title("Row 100 intensity profile", fontsize=9)
    axes[1, col].set_xlabel("Column")
    axes[1, col].set_ylabel("Pixel value")

plt.suptitle("Demo 1: Point Operations - Each Output Pixel Depends ONLY On Corresponding Input Pixel", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_point_ops.png", dpi=150, bbox_inches="tight")
print("[Saved] -> demo_point_ops.png")
plt.close()

# ============================================================
# Demo 2: Mean filter vs Median filter on salt-pepper noise
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Mean filter vs Median filter (salt-pepper noise)")
print("=" * 60)

clear_img = np.zeros((300, 400), dtype=np.uint8)
clear_img[:, :] = np.linspace(0, 255, 400, dtype=np.uint8)[np.newaxis, :]
clear_img[100:200, 50:350] = 180
clear_img[120:180, 80:320] = 80

noisy_img = clear_img.copy()
num_salt = int(clear_img.size * 0.05)
num_pepper = int(clear_img.size * 0.05)
salt_coords = (np.random.randint(0, clear_img.shape[0], num_salt), np.random.randint(0, clear_img.shape[1], num_salt))
pepper_coords = (np.random.randint(0, clear_img.shape[0], num_pepper), np.random.randint(0, clear_img.shape[1], num_pepper))
noisy_img[salt_coords] = 255
noisy_img[pepper_coords] = 0

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

kernel_sizes = [3, 5, 7]
mean_results = {k: cv2.filter2D(noisy_img, -1, np.ones((k, k), dtype=np.float32) / (k * k)) for k in kernel_sizes}
median_results = {k: cv2.medianBlur(noisy_img, k) for k in kernel_sizes}

print("\n[PSNR comparison (higher=better, inf=identical)]")
print(f"{'Method':>20} | {'k=3':>8} | {'k=5':>8} | {'k=7':>8}")
print("-" * 55)
for k in kernel_sizes:
    p_mean = psnr(clear_img, mean_results[k])
    p_med = psnr(clear_img, median_results[k])
    print(f"{'Mean k='+str(k):>20} | {p_mean:>8.1f} | {psnr(clear_img, cv2.filter2D(noisy_img, -1, np.ones((5,5),dtype=np.float32)/25)):>8.1f} | {psnr(clear_img, cv2.filter2D(noisy_img,-1,np.ones((7,7),dtype=np.float32)/49)):>8.1f}")
    print(f"{'Median k='+str(k):>20} | {psnr(clear_img,median_results[3]):>8.1f} | {psnr(clear_img,median_results[5]):>8.1f} | {psnr(clear_img,median_results[7]):>8.1f}")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))

titles_and_imgs = [
    ("Original (no noise)", clear_img),
    ("Salt-Pepper 10%", noisy_img),
    ("Mean k=3", mean_results[3]),
    ("Mean k=5", mean_results[5]),
    ("Median k=3", median_results[3]),
]

for col, (title, img) in enumerate(titles_and_imgs):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

crop_y, crop_x = 120, 80
crop_size = 80
for col, (title, img) in enumerate(titles_and_imgs):
    crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size*2]
    axes[1, col].imshow(crop, cmap="gray")
    axes[1, col].set_title(f"{title} (zoom)", fontsize=9)
    axes[1, col].axis("off")

axes[2, 0].imshow(mean_results[7], cmap="gray")
axes[2, 0].set_title("Mean k=7 (over-blurred)", fontsize=10)
axes[2, 0].axis("off")

axes[2, 1].imshow(median_results[7], cmap="gray")
axes[2, 1].set_title("Median k=7 (clean, preserves edges)", fontsize=10)
axes[2, 1].axis("off")

methods = ["Mean k=3", "Median k=3", "Mean k=5", "Median k=5", "Mean k=7", "Median k=7"]
psnr_values = [psnr(clear_img, mean_results[3]), psnr(clear_img, median_results[3]),
               psnr(clear_img, mean_results[5]), psnr(clear_img, median_results[5]),
               psnr(clear_img, mean_results[7]), psnr(clear_img, median_results[7])]
colors = ["#ff7f7f", "#7fbf7f", "#ff7f7f", "#7fbf7f", "#ff7f7f", "#7fbf7f"]
axes[2, 2].bar(methods, psnr_values, color=colors)
axes[2, 2].set_ylabel("PSNR (dB)")
axes[2, 2].set_title("Quality comparison (higher=better)")
axes[2, 2].tick_params(axis='x', rotation=45)
axes[2, 3].axis("off")
axes[2, 4].axis("off")

plt.suptitle("Demo 2: Mean Filter vs Median Filter on Salt-Pepper Noise\n(Median ignores extreme values -> naturally removes impulse noise)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_filters.png", dpi=150, bbox_inches="tight")
print("[Saved] -> demo_filters.png")
plt.close()

# ============================================================
# Demo 3: Superposition principle verification
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Superposition principle (linear vs nonlinear)")
print("=" * 60)

f = np.zeros((50, 50), dtype=np.float32)
f[15:35, 15:35] = 200
g = np.zeros((50, 50), dtype=np.float32)
g[5:15, 5:15] = 100
f_plus_g = f + g

def mean_filter(img, k=3):
    return cv2.filter2D(img, -1, np.ones((k, k)) / (k * k))

def median_filter(img, k=3):
    return cv2.medianBlur(img.astype(np.uint8), k).astype(np.float32)

T_f = mean_filter(f, k=3)
T_g = mean_filter(g, k=3)
T_f_plus_g = mean_filter(f_plus_g, k=3)
T_f_plus_T_g = T_f + T_g
linear_error = np.mean(np.abs(T_f_plus_g - T_f_plus_T_g))
print(f"[Linear (Mean filter)] T[f+g] vs T[f]+T[g] MAE: {linear_error:.6f}  {'-> Superposition holds' if linear_error < 1e-5 else '-> FAILED'}")

T_f_nl = median_filter(f, k=3)
T_g_nl = median_filter(g, k=3)
T_f_plus_g_nl = median_filter(f_plus_g, k=3)
T_f_plus_T_g_nl = T_f_nl + T_g_nl
nonlinear_error = np.mean(np.abs(T_f_plus_g_nl - T_f_plus_T_g_nl))
print(f"[Nonlinear (Median)] T[f+g] vs T[f]+T[g] MAE: {nonlinear_error:.2f}  {'-> Superposition holds' if nonlinear_error < 1e-5 else '-> Superposition FAILS (expected)'}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=200)
axes[0, 0].set_title("f (center block)")
axes[0, 0].axis("off")

axes[0, 1].imshow(g, cmap="gray", vmin=0, vmax=200)
axes[0, 1].set_title("g (corner block)")
axes[0, 1].axis("off")

axes[0, 2].imshow(f_plus_g, cmap="gray", vmin=0, vmax=300)
axes[0, 2].set_title("f + g")
axes[0, 2].axis("off")

axes[0, 3].imshow(T_f_plus_T_g - T_f_plus_g, cmap="gray")
axes[0, 3].set_title(f"Linear error: {linear_error:.6f}\n(~0 = superposition holds)")
axes[0, 3].axis("off")

axes[1, 0].imshow(T_f, cmap="gray")
axes[1, 0].set_title("T[f] (mean filter)")
axes[1, 0].axis("off")

axes[1, 1].imshow(T_g, cmap="gray")
axes[1, 1].set_title("T[g] (mean filter)")
axes[1, 1].axis("off")

axes[1, 2].imshow(T_f_plus_g_nl, cmap="gray")
axes[1, 2].set_title("T[f+g] (median filter)")
axes[1, 2].axis("off")

axes[1, 3].imshow(T_f_plus_T_g_nl - T_f_plus_g_nl, cmap="gray")
axes[1, 3].set_title(f"Nonlinear error: {nonlinear_error:.1f}\n(non-zero = superposition fails)")
axes[1, 3].axis("off")

plt.suptitle("Demo 3: Superposition Principle - Linear vs Nonlinear\n(Mean filter=linear, Median filter=nonlinear)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_superposition.png", dpi=150, bbox_inches="tight")
print("[Saved] -> demo_superposition.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Point operations: each output pixel depends ONLY on corresponding input pixel (LUT-acceleratable)")
print("2. Mean filter: linear (satisfies superposition), but blurs noise into neighbors")
print("3. Median filter: nonlinear (fails superposition), naturally ignores extreme values -> best for impulse noise")
print("4. Salt-pepper noise -> Median filter; Gaussian noise -> Mean/Gaussian filter")
print("5. Linear ops can be accelerated via frequency domain (nonlinear cannot)")
