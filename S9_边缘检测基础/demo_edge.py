"""
Chapter 9: Edge Detection and Segmentation - Demo
===================================================
Goals:
  1. Compare Roberts / Sobel / Prewitt edge detectors
  2. Understand gradient magnitude and direction
  3. Verify Sobel sensitivity to noise -> Gaussian smoothing is necessary
  4. Compare fixed / Otsu / adaptive thresholding
  5. Understand why Otsu fails on unimodal histograms

Install:
  pip install opencv-python numpy matplotlib

Run:
  python demo_edge.py
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
# Demo 1: Three differential operators
# ============================================================
print("=" * 60)
print("Demo 1: Roberts / Sobel / Prewitt edge detector comparison")
print("=" * 60)

test_gray = np.zeros((300, 400), dtype=np.uint8)
test_gray[:, 100:110] = 200
test_gray[:, 200:220] = 100
test_gray[80:90, :] = 200
test_gray[180:200, :] = 50

roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
G_x = cv2.filter2D(test_gray.astype(np.float32), -1, roberts_x)
G_y = cv2.filter2D(test_gray.astype(np.float32), -1, roberts_y)
roberts = np.clip(np.sqrt(G_x**2 + G_y**2), 0, 255).astype(np.uint8)

sobel_x = cv2.Sobel(test_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(test_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255).astype(np.uint8)

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
P_x = cv2.filter2D(test_gray.astype(np.float32), -1, prewitt_x)
P_y = cv2.filter2D(test_gray.astype(np.float32), -1, prewitt_y)
prewitt = np.clip(np.sqrt(P_x**2 + P_y**2), 0, 255).astype(np.uint8)

def edge_strength(img):
    return np.mean(img), np.max(img), np.sum(img > 50)

print(f"\n[Edge strength]")
print(f"{'Operator':>10} | {'Mean':>8} | {'Max':>8} | {'Strong edges':>12}")
print("-" * 45)
for name, img in [("Roberts", roberts), ("Sobel", sobel), ("Prewitt", prewitt)]:
    avg, mx, strong = edge_strength(img)
    print(f"{name:>10} | {avg:>8.1f} | {mx:>8.1f} | {strong:>12}")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(test_gray, cmap="gray")
axes[0, 0].set_title("Original (gray)")
axes[0, 0].axis("off")

operators = [
    ("Roberts\n(2x2, diagonal diff)", roberts),
    ("Prewitt\n(3x3, uniform smooth)", prewitt),
    ("Sobel\n(3x3, weighted smooth [1,2,1])", sobel),
]

for col, (title, img) in enumerate(operators, 1):
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

crop_y, crop_x = 100, 80
crop_size = 80

axes[1, 0].imshow(test_gray[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 0].set_title("Original (zoom)")
axes[1, 0].axis("off")

for col, (title, img) in enumerate(operators, 1):
    crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    axes[1, col].imshow(crop, cmap="gray")
    axes[1, col].set_title(f"{title.split(chr(10))[0]} (zoom)", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("Demo 1: Three Differential Operators\n(Roberts=thinnest edges but noisy, Sobel=most commonly used)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_operators.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_operators.png")
plt.close()

# ============================================================
# Demo 2: Noise -> edge detection degradation -> Gaussian smooth helps
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Noise -> edge detection degradation -> Gaussian smooth helps")
print("=" * 60)

noise = np.random.normal(0, 25, test_gray.shape).astype(np.int16)
noisy = np.clip(test_gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

sobel_noisy = np.clip(cv2.Sobel(noisy, cv2.CV_64F, 1, 0, ksize=3), 0, 255).astype(np.uint8)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.5)
sobel_smooth = np.clip(cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3), 0, 255).astype(np.uint8)
mean_blur = cv2.blur(noisy, (5, 5))
sobel_mean = np.clip(cv2.Sobel(mean_blur, cv2.CV_64F, 1, 0, ksize=3), 0, 255).astype(np.uint8)

gt_edges = np.clip(cv2.Sobel(test_gray, cv2.CV_64F, 1, 0, ksize=3), 0, 255).astype(np.uint8)

def edge_f1(edges, gt):
    edges_binary = (edges > 50).astype(float)
    gt_binary = (gt > 50).astype(float)
    intersection = np.sum(edges_binary * gt_binary)
    precision = intersection / (np.sum(edges_binary) + 1e-6)
    recall = intersection / (np.sum(gt_binary) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

p1, r1, f1_1 = edge_f1(sobel_noisy, gt_edges)
p2, r2, f1_2 = edge_f1(sobel_smooth, gt_edges)
p3, r3, f1_3 = edge_f1(sobel_mean, gt_edges)

print(f"\n[Edge quality (F1 score, higher=better)]")
print(f"{'Method':>20} | {'Precision':>10} | {'Recall':>10} | {'F1':>8}")
print("-" * 55)
print(f"{'Sobel (no preprocessing)':>20} | {p1:>10.3f} | {r1:>10.3f} | {f1_1:>8.3f}")
print(f"{'Gaussian smooth + Sobel':>20} | {p2:>10.3f} | {r2:>10.3f} | {f1_2:>8.3f}")
print(f"{'Mean smooth + Sobel':>20} | {p3:>10.3f} | {r3:>10.3f} | {f1_3:>8.3f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(test_gray, cmap="gray")
axes[0, 0].set_title("Original (no noise)")
axes[0, 0].axis("off")

axes[0, 1].imshow(noisy, cmap="gray")
axes[0, 1].set_title("Gaussian noise added (sigma=25)")
axes[0, 1].axis("off")

axes[0, 2].imshow(gaussian, cmap="gray")
axes[0, 2].set_title("Gaussian smooth (sigma=1.5)")
axes[0, 2].axis("off")

axes[1, 0].imshow(sobel_noisy, cmap="gray")
axes[1, 0].set_title(f"Direct Sobel (F1={f1_1:.2f} - noise=edge)")
axes[1, 0].axis("off")

axes[1, 1].imshow(sobel_mean, cmap="gray")
axes[1, 1].set_title(f"Mean smooth + Sobel (F1={f1_3:.2f})")
axes[1, 1].axis("off")

axes[1, 2].imshow(sobel_smooth, cmap="gray")
axes[1, 2].set_title(f"Gaussian + Sobel (F1={f1_2:.2f} - best)")
axes[1, 2].axis("off")

plt.suptitle("Demo 2: Noise -> Edge Detection Degradation -> Gaussian Smooth is the Solution\n(Sobel amplifies high frequencies -> amplifies noise -> must smooth first)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_noise.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_noise.png")
plt.close()

# ============================================================
# Demo 3: Fixed / Otsu / Adaptive thresholding
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Fixed / Otsu / Adaptive thresholding")
print("=" * 60)

bimodal = np.zeros((200, 400), dtype=np.uint8)
bimodal[:, :200] = 60 + np.random.normal(0, 8, (200, 200)).astype(np.int16)
bimodal[:, 200:] = 200 + np.random.normal(0, 8, (200, 200)).astype(np.int16)
bimodal = np.clip(bimodal, 0, 255).astype(np.uint8)

unimodal = np.zeros((200, 400), dtype=np.uint8)
unimodal[:] = 128 + np.random.normal(0, 25, (200, 400)).astype(np.int16)
unimodal = np.clip(unimodal, 0, 255).astype(np.uint8)

uneven = np.zeros((200, 400), dtype=np.uint8)
for i in range(400):
    brightness = int(80 + i * 0.4)
    noise_val = np.random.normal(0, 10)
    uneven[:, i] = np.clip(brightness + noise_val, 0, 255).astype(np.uint8)
uneven[80:120, 100:300] = np.clip(uneven[80:120, 100:300] + 80, 0, 255)

scenarios = [
    ("Bimodal\n(target/bg separated)", bimodal),
    ("Unimodal\n(no clear threshold)", unimodal),
    ("Uneven illumination\n(left dark, right bright)", uneven),
]

fig, axes = plt.subplots(len(scenarios), 5, figsize=(20, 12))

for row, (name, img) in enumerate(scenarios):
    axes[row, 0].imshow(img, cmap="gray")
    axes[row, 0].set_title(name, fontsize=10)
    axes[row, 0].axis("off")

    _, fixed = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    axes[row, 1].imshow(fixed, cmap="gray")
    axes[row, 1].set_title("Fixed T=128", fontsize=9)
    axes[row, 1].axis("off")

    if row < 2:
        T_otsu, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        axes[row, 2].imshow(otsu, cmap="gray")
        axes[row, 2].set_title(f"Otsu T={T_otsu:.0f}", fontsize=9)
        axes[row, 2].axis("off")
    else:
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].set_title("Otsu fails\n(unimodal histogram)", fontsize=9)
        axes[row, 2].axis("off")

    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    axes[row, 3].imshow(adaptive, cmap="gray")
    axes[row, 3].set_title("Adaptive threshold\n(GAUSSIAN_C)", fontsize=9)
    axes[row, 3].axis("off")

    axes[row, 4].hist(img.ravel(), bins=50, color="gray", alpha=0.7)
    axes[row, 4].axvline(x=128, color="b", linestyle="--", label="T=128")
    if row < 2:
        T_otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        axes[row, 4].axvline(x=T_otsu, color="r", linestyle="--", label=f"Otsu T={T_otsu:.0f}")
    axes[row, 4].set_title("Histogram", fontsize=9)
    axes[row, 4].legend(fontsize=7)

plt.suptitle("Demo 3: Thresholding Methods\n(Bimodal->Otsu works | Unimodal->Otsu fails | Uneven light->Adaptive threshold)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_thresholding.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_thresholding.png")
plt.close()

T_uni = cv2.threshold(unimodal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
print(f"\n[Otsu on unimodal] T={T_uni:.0f} -> meaningless (no two classes in unimodal distribution)")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Roberts/Prewitt/Sobel are all differential operators (differences: kernel size and weights)")
print("2. Differential -> amplifies high frequencies -> amplifies noise -> must smooth first (Gaussian>Mean)")
print("3. Fixed threshold: fast but needs manual tuning; Otsu: automatic but needs bimodal distribution")
print("4. Adaptive threshold: local T per region -> best for uneven illumination")
print("5. Otsu fails on unimodal histograms -> consider entropy-based methods instead")
