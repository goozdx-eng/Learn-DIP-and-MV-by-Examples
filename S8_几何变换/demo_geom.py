"""
Chapter 8: Geometric Transformations - Demo
==========================================
Goals:
  1. Compare nearest-neighbor, bilinear, bicubic interpolation
  2. Observe interpolation error accumulation from repeated transforms
  3. Understand image registration 3-step process
  4. Verify resampling necessity before registration

Install:
  pip install opencv-python numpy matplotlib scikit-image

Run:
  python demo_geom.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

import cv2
import numpy as np
import matplotlib.pyplot as plt
try:
    from skimage import data
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

_cjk_fonts = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name.lower() for k in ['noto', 'wqy', 'simsun', 'simhei', 'microsoft yahei', 'pingfang', 'heiti'])]
if _cjk_fonts:
    plt.rcParams['font.family'] = _cjk_fonts[0]
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Demo 1: Three interpolation methods
# ============================================================
print("=" * 60)
print("Demo 1: Nearest-neighbor / Bilinear / Bicubic interpolation")
print("=" * 60)

test_img = np.zeros((300, 400), dtype=np.uint8)
test_img[:, :] = 200
for i in range(300):
    j = int(i * 1.5)
    if j < 400:
        test_img[i, j] = 50
test_img[::20, :] = 100
test_img[:, ::20] = 100

scale = 0.5
H, W = test_img.shape

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

interpolations = {
    "Nearest (INTER_NEAREST)": cv2.INTER_NEAREST,
    "Bilinear (INTER_LINEAR)": cv2.INTER_LINEAR,
    "Bicubic (INTER_CUBIC)": cv2.INTER_CUBIC,
}

results = {}
for name, interp in interpolations.items():
    small = cv2.resize(test_img, (int(W * scale), int(H * scale)), interpolation=interp)
    restored = cv2.resize(small, (W, H), interpolation=interp)
    results[name] = restored

print("\n[Interpolation quality (PSNR, higher=better)]")
for name, img in results.items():
    p = psnr(test_img, img)
    print(f"  {name}: PSNR={p:.1f} dB")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(test_img, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for col, (name, img) in enumerate(results.items(), 1):
    axes[0, col].imshow(img, cmap="gray")
    p = psnr(test_img, img)
    axes[0, col].set_title(f"{name}\nPSNR={p:.1f}dB")
    axes[0, col].axis("off")

crop_y, crop_x = 50, 100
crop_size = 100

axes[1, 0].imshow(test_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap="gray")
axes[1, 0].set_title("Original (zoom)")
axes[1, 0].axis("off")

for col, (name, img) in enumerate(results.items(), 1):
    crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    axes[1, col].imshow(crop, cmap="gray")
    labels = ["Nearest (jagged edges)", "Bilinear (smooth)", "Bicubic (smoothest)"]
    axes[1, col].set_title(f"{labels[col-1]}")
    axes[1, col].axis("off")

plt.suptitle("Demo 1: Three Interpolation Methods\n(Scale 50% then restore -> observe diagonal edge quality)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_interpolation.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_interpolation.png")
plt.close()

# ============================================================
# Demo 2: Interpolation error accumulation
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: Interpolation error accumulation (repeated rotation)")
print("=" * 60)

angle = 1.0
center = (W // 2, H // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)

current = test_img.copy()
psnr_history = [float('inf')]

print("\n[Rotation 10 times (1 degree each) -> PSNR degradation]")
for i in range(10):
    current = cv2.warpAffine(current, M, (W, H), flags=cv2.INTER_LINEAR)
    p = psnr(test_img, current)
    psnr_history.append(p)
    flag = " [Blurry]" if p < 25 else ""
    print(f"  Iteration {i+1:2d}: PSNR={p:.1f} dB{flag}")

plt.figure(figsize=(10, 4))
plt.plot(range(11), psnr_history, "bo-", linewidth=2, markersize=8)
plt.axhline(y=30, color="r", linestyle="--", label="PSNR=30dB (visible degradation)")
plt.xlabel("Rotation iterations (1 degree each)")
plt.ylabel("PSNR (dB)")
plt.title("Demo 2: Repeated Rotation -> Interpolation Error Accumulation\n(Each rotation estimates sub-pixel coords -> error accumulates)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("demo_accumulation.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_accumulation.png")
plt.close()

# ============================================================
# Demo 3: Image registration (feature extraction -> match -> transform)
# ============================================================
print("\n" + "=" * 60)
print("Demo 3: Image Registration (feature -> match -> transform)")
print("=" * 60)

if HAS_SKIMAGE:
    ref_img = data.page()
    # skimage data.page() can be grayscale or RGB
    if len(ref_img.shape) == 2:
        ref_gray = cv2.resize(ref_img, (400, 300))
    else:
        ref_gray = cv2.resize(cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY), (400, 300))
    ref_gray = (ref_gray / ref_gray.max() * 255).astype(np.uint8)
else:
    # Synthetic document-like image (text-like edges and structure)
    ref_gray = np.full((300, 400), 220, dtype=np.uint8)
    # Add text-like horizontal lines
    ref_gray[30:34, :] = 40
    ref_gray[70:74, :] = 40
    ref_gray[110:114, :] = 40
    ref_gray[150:154, :] = 40
    ref_gray[190:194, :] = 40
    # Add vertical lines (characters)
    for x in range(20, 380, 25):
        ref_gray[30:200, x:x+2] = 40
    # Add some gray blocks (paragraphs)
    ref_gray[60:100, 50:350] = 180
    ref_gray[120:160, 50:350] = 160

M_offset = np.float32([[1.0, 0.0, 30], [0.0, 1.0, 20]])
offset_img = cv2.warpAffine(ref_gray, M_offset, (400, 300))

angle = 5.0
M_rotate = cv2.getRotationMatrix2D((W//2, H//2), angle, 1.0)
M_combined = np.dot(M_rotate, np.float32([[1, 0, 30], [0, 1, 20], [0, 0, 1]]))[:2, :]
moved_img = cv2.warpAffine(ref_gray, M_combined, (400, 300))

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(ref_gray, None)
kp2, des2 = orb.detectAndCompute(moved_img, None)

print(f"  Reference features: {len(kp1)}")
print(f"  Moved image features: {len(kp2)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]

print(f"  Matches: {len(matches)}, top {len(good_matches)} used")

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
inliers = mask.ravel().sum()
print(f"  RANSAC inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.0f}%)")

aligned = cv2.warpPerspective(ref_gray, M_est, (400, 300))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(ref_gray, cmap="gray")
axes[0, 0].set_title("Reference image")
axes[0, 0].axis("off")

axes[0, 1].imshow(moved_img, cmap="gray")
axes[0, 1].set_title("Moved image (translated + rotated)")
axes[0, 1].axis("off")

axes[0, 2].imshow(aligned, cmap="gray")
axes[0, 2].set_title("Registered (aligned)")
axes[0, 2].axis("off")

match_img = cv2.drawMatches(ref_gray, kp1, moved_img, kp2, good_matches[:20], None)
axes[1, 0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Feature matches (top 20)")
axes[1, 0].axis("off")

error = np.abs(ref_gray.astype(float) - aligned.astype(float))
im = axes[1, 1].imshow(error, cmap="hot")
axes[1, 1].set_title("Registration error (absolute difference)")
axes[1, 1].axis("off")
plt.colorbar(im, ax=axes[1, 1])

overlay = np.zeros((300, 400, 3), dtype=np.uint8)
overlay[:, :, 0] = ref_gray
overlay[:, :, 2] = aligned
axes[1, 2].imshow(overlay)
axes[1, 2].set_title("Overlay (red=ref, blue=aligned)\n(Good overlap=yellow)")
axes[1, 2].axis("off")

plt.suptitle("Demo 3: Image Registration 3-Step Process\n(Feature extraction -> Feature matching -> RANSAC transform estimation)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("demo_registration.png", dpi=150, bbox_inches="tight")
print("\n[Saved] -> demo_registration.png")
plt.close()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("1. Nearest=jagged, Bilinear=balanced, Bicubic=smoothest but slowest")
print("2. Repeated transforms accumulate interpolation error -> minimize transform count")
print("3. Registration: Feature extraction(ORB/SIFT) -> Feature matching(BFMatcher) -> RANSAC transform estimation")
print("4. Registration requires spatial resolution alignment (resampling) first")
