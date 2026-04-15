"""
Chapter 7: Histogram - Demo
===========================
Goals:
  1. Observe histogram shapes of different image types (low contrast, bimodal, overexposed, underexposed)
  2. Verify histogram equalization spreads concentrated gray levels
  3. Understand CDF mapping mechanism
  4. Observe equalization pitfalls (noise amplification, uneven illumination)
  5. Understand histogram specification (target shape matching)

Install:
  pip install opencv-python numpy matplotlib

Run:
  python demo_histogram.py
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
# Demo 1: Different histogram shapes
# ============================================================
print("=" * 60)
print("Demo 1: Four typical histogram shapes")
print("=" * 60)

# Low contrast (gray levels concentrated in middle)
low_contrast = np.zeros((200, 400), dtype=np.uint8)
mu, sigma = 150, 30
x = np.arange(400)
y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
y = (y / y.max() * 80 + 80).astype(int)
for i in range(200):
    low_contrast[i, :] = y

# Bimodal (target and background separated)
bimodal = np.zeros((200, 400), dtype=np.uint8)
bimodal[:, :200] = 60
bimodal[:, 200:] = 200
noise = np.random.normal(0, 10, bimodal.shape).astype(np.int16)
bimodal = np.clip(bimodal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Overexposed (gray levels concentrated at top)
overexposed = np.full((200, 400), 200, dtype=np.uint8)
overexposed[:, 50:350] = np.linspace(180, 255, 300, dtype=np.uint8)
noise_over = np.random.normal(0, 5, overexposed.shape).astype(np.int16)
overexposed = np.clip(overexposed.astype(np.int16) + noise_over, 0, 255).astype(np.uint8)

# Underexposed (gray levels concentrated at bottom)
underexposed = np.full((200, 400), 55, dtype=np.uint8)
underexposed[:, 50:350] = np.linspace(20, 120, 300, dtype=np.uint8)
noise_under = np.random.normal(0, 5, underexposed.shape).astype(np.int16)
underexposed = np.clip(underexposed.astype(np.int16) + noise_under, 0, 255).astype(np.uint8)

def compute_hist(img, bins=256):
    return cv2.calcHist([img], [0], None, [bins], [0, bins]).flatten()

hist_low = compute_hist(low_contrast)
hist_bimodal = compute_hist(bimodal)
hist_over = compute_hist(overexposed)
hist_under = compute_hist(underexposed)

for name, hist in [("Low contrast", hist_low), ("Bimodal", hist_bimodal),
                    ("Overexposed", hist_over), ("Underexposed", hist_under)]:
    nonzero = np.nonzero(hist)[0]
    if len(nonzero) > 0:
        print(f"  {name}: range=[{nonzero.min()}, {nonzero.max()}], peak at gray={np.argmax(hist)}")

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
scenarios = [
    ("Low contrast\n(gray concentrated in middle)", low_contrast, hist_low),
    ("Bimodal\n(target/background separated)", bimodal, hist_bimodal),
    ("Overexposed\n(gray concentrated at top)", overexposed, hist_over),
    ("Underexposed\n(gray concentrated at bottom)", underexposed, hist_under),
]

for row, (title, img, hist) in enumerate(scenarios):
    axes[row, 0].imshow(img, cmap="gray")
    axes[row, 0].set_title(title, fontsize=11)
    axes[row, 0].axis("off")
    axes[row, 1].fill_between(range(256), hist, color="gray", alpha=0.7)
    axes[row, 1].set_xlim(0, 255)
    axes[row, 1].set_title(f"Histogram (peak={np.argmax(hist)})", fontsize=10)
    axes[row, 1].set_xlabel("Gray value")
    axes[row, 1].set_ylabel("Pixel count")

plt.suptitle("Demo 1: Four Typical Histogram Shapes\n(HISTOGRAM = distribution, not image - it tells you the image's 'personality')", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_shapes.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_shapes.png")
plt.close()

# ============================================================
# Demo 2: Histogram equalization (CDF mapping)
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Histogram equalization (CDF mapping)")
print("=" * 60)

gray_low = low_contrast.copy()
eq_img = cv2.equalizeHist(gray_low)

def manual_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    cdf_mapped = np.round(cdf_normalized * 255).astype(np.uint8)
    return cdf_mapped[img], cdf_normalized

manual_eq, cdf = manual_equalize(gray_low)

hist_before = compute_hist(gray_low)
hist_after = compute_hist(eq_img)
before_range = np.nonzero(hist_before)[0]
after_range = np.nonzero(hist_after)[0]

print(f"[Equalization] Before: range=[{before_range.min()}, {before_range.max()}]")
print(f"[Equalization] After:  range=[{after_range.min()}, {after_range.max()}]")
print(f"[Equalization] Manual matches OpenCV: {np.array_equal(eq_img, manual_eq)}")

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

axes[0, 0].imshow(gray_low, cmap="gray")
axes[0, 0].set_title("Original (low contrast)")
axes[0, 0].axis("off")

axes[0, 1].fill_between(range(256), hist_before, color="gray", alpha=0.7)
axes[0, 1].set_title("Original histogram")
axes[0, 1].set_xlim(0, 255)

axes[0, 2].plot(range(256), cdf, color="blue", linewidth=2)
axes[0, 2].set_title("CDF (cumulative distribution)")
axes[0, 2].set_xlabel("Gray value")
axes[0, 2].set_ylabel("Cumulative probability")
axes[0, 2].set_xlim(0, 255)

axes[1, 0].imshow(eq_img, cmap="gray")
axes[1, 0].set_title("After equalization (contrast enhanced)")
axes[1, 0].axis("off")

axes[1, 1].fill_between(range(256), hist_after, color="gray", alpha=0.7)
axes[1, 1].set_title("Equalized histogram")
axes[1, 1].set_xlim(0, 255)

axes[1, 2].plot(range(256), cdf * 255, color="red", linewidth=2)
axes[1, 2].set_title("Gray mapping function (old->new)")
axes[1, 2].set_xlabel("Original gray")
axes[1, 2].set_ylabel("New gray")
axes[1, 2].set_xlim(0, 255)
axes[1, 2].set_ylim(0, 255)

axes[2, 0].imshow(gray_low, cmap="gray")
axes[2, 0].set_title("Before equalization")
axes[2, 0].axis("off")

axes[2, 1].imshow(eq_img, cmap="gray")
axes[2, 1].set_title("After equalization")
axes[2, 1].axis("off")

axes[2, 2].plot(range(256), hist_before, color="blue", alpha=0.7, label="Before")
axes[2, 2].plot(range(256), hist_after, color="red", alpha=0.7, label="After")
axes[2, 2].set_title("Histogram comparison")
axes[2, 2].legend()
axes[2, 2].set_xlim(0, 255)

plt.suptitle("Demo 2: Histogram Equalization via CDF Mapping\n(CDF maps concentrated [60,160] to full [0,255] range)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_equalization.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_equalization.png")
plt.close()

# ============================================================
# Demo 3: Equalization pitfalls (noise + uneven illumination)
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Equalization pitfalls (noise amplification + uneven light)")
print("=" * 60)

flat_noise = np.full((200, 400), 128, dtype=np.uint8)
flat_noise[:, 50:350] = 120
flat_noise[::5, ::5] = 0
flat_noise[2::5, 2::5] = 255

uneven_light = np.zeros((200, 400), dtype=np.uint8)
for i in range(400):
    brightness = int(255 * (0.3 + 0.7 * i / 400))
    uneven_light[:, i] = brightness
uneven_light[80:120, :] = np.clip(uneven_light[80:120, :] + 80, 0, 255)

flat_eq = cv2.equalizeHist(flat_noise)
uneven_eq = cv2.equalizeHist(uneven_light)

def measure_noise_variance(img, roi):
    y0, x0, h, w = roi
    return np.var(img[y0:y0+h, x0:x0+w])

flat_noise_before = measure_noise_variance(flat_noise, (0, 0, 50, 50))
flat_noise_after = measure_noise_variance(flat_eq, (0, 0, 50, 50))
ratio = flat_noise_after / flat_noise_before
print(f"[Noise amplification] Before: {flat_noise_before:.2f}, After: {flat_noise_after:.2f}, Ratio: {ratio:.1f}x")
print(f"[Warning] Noise was amplified {ratio:.1f}x!" if ratio > 2 else "[OK] Noise not significantly amplified")

left_before = np.mean(uneven_light[:, :50])
right_before = np.mean(uneven_light[:, 350:])
left_after = np.mean(uneven_eq[:, :50])
right_after = np.mean(uneven_eq[:, 350:])
print(f"[Uneven light] Before: L={left_before:.0f} R={right_before:.0f} diff={abs(left_before-right_before):.0f}")
print(f"[Uneven light] After:  L={left_after:.0f} R={right_after:.0f} diff={abs(left_after-right_after):.0f}")
print(f"[Warning] Left side overexposed!" if left_after > 200 else "[OK] No overexposure")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(flat_noise, cmap="gray")
axes[0, 0].set_title("Original (flat + noise)")
axes[0, 0].axis("off")

axes[0, 1].imshow(flat_eq, cmap="gray")
axes[0, 1].set_title("After equalization (noise AMPLIFIED)")
axes[0, 1].axis("off")

axes[0, 2].hist(flat_noise.ravel(), bins=50, alpha=0.7, label="Before")
axes[0, 2].hist(flat_eq.ravel(), bins=50, alpha=0.7, label="After")
axes[0, 2].set_title("Histogram comparison")
axes[0, 2].legend()

axes[0, 3].imshow(flat_noise[:100, :100], cmap="gray")
axes[0, 3].set_title("Noise detail (zoom)")
axes[0, 3].axis("off")

axes[1, 0].imshow(uneven_light, cmap="gray")
axes[1, 0].set_title("Original (left dark, right bright)")
axes[1, 0].axis("off")

axes[1, 1].imshow(uneven_eq, cmap="gray")
axes[1, 1].set_title("After equalization (left OVEREXPOSED)")
axes[1, 1].axis("off")

axes[1, 2].plot(range(400), [np.mean(uneven_light[:, i]) for i in range(400)], label="Before")
axes[1, 2].plot(range(400), [np.mean(uneven_eq[:, i]) for i in range(400)], label="After")
axes[1, 2].set_title("Brightness per column")
axes[1, 2].legend()

axes[1, 3].axis("off")

plt.suptitle("Demo 3: Equalization Pitfalls\n(Top: noise amplified | Bottom: uneven illumination -> overexposure)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_pitfalls.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_pitfalls.png")
plt.close()

# ============================================================
# Demo 4: CLAHE (adaptive histogram equalization)
# ============================================================
print("\n" + "=" * 60)
print("Demo 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)")
print("=" * 60)

clahe_test = uneven_light.copy()
clahe_test[80:120, :] = np.clip(clahe_test[80:120, :] + 80, 0, 255)
global_eq = cv2.equalizeHist(clahe_test)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_eq = clahe.apply(clahe_test)

print(f"[CLAHE vs Global] Global: L={np.mean(global_eq[:,50]):.0f} R={np.mean(global_eq[:,350]):.0f}")
print(f"[CLAHE vs Global] CLAHE:  L={np.mean(clahe_eq[:,50]):.0f} R={np.mean(clahe_eq[:,350]):.0f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(clahe_test, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(global_eq, cmap="gray")
axes[1].set_title("Global equalization\n(left overexposed)")
axes[1].axis("off")

axes[2].imshow(clahe_eq, cmap="gray")
axes[2].set_title("CLAHE (adaptive)\n(local contrast preserved)")
axes[2].axis("off")

axes[3].plot([np.mean(clahe_test[:, i]) for i in range(400)], label="Original")
axes[3].plot([np.mean(global_eq[:, i]) for i in range(400)], label="Global eq")
axes[3].plot([np.mean(clahe_eq[:, i]) for i in range(400)], label="CLAHE")
axes[3].legend()
axes[3].set_title("Brightness curve comparison")

plt.suptitle("Demo 4: CLAHE (Adaptive Histogram Equalization)\n(Divides image into tiles, equalizes each -> no global overexposure)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_clahe.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_clahe.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Histogram = distribution (no spatial info), but reveals image's state quickly")
print("2. Equalization uses CDF to spread concentrated gray levels -> enhances contrast")
print("3. Equalization pitfalls: amplifies noise, overexposes unevenly-lit images")
print("4. CLAHE (adaptive equalization) solves uneven illumination -> standard in medical/satellite imaging")
