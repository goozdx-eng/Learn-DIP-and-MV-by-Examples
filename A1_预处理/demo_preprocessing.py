"""
Stage 1: Image Preprocessing - OpenCV Examples
Covers: Point operations, Histogram equalization, CLAHE, Affine transform, Perspective transform

Install: pip install opencv-python numpy matplotlib
Run: python demo_preprocessing.py
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Utility: Unified multi-image display
# ============================================================
def show_images(images, titles, cmap_list=None, figsize=(16, 4), save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title, cmap in zip(axes, images, titles, cmap_list or ['gray'] * n):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] -> {save_path}")
    plt.close()


# ============================================================
# 1. Load test image (auto-generate synthetic plate if none)
# ============================================================
def create_fake_plate(shape=(120, 400)):
    """Generate synthetic low-quality license plate: overall dark + slight rotation"""
    img = np.zeros(shape, dtype=np.uint8)
    img[:] = 40  # dark background
    positions = [20, 70, 120, 170, 230, 280, 330]
    for x in positions:
        cv2.rectangle(img, (x, 20), (x + 40, 100), 200, -1)
    noise = np.random.randint(0, 50, shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    M = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), 8, 1.0)
    img = cv2.warpAffine(img, M, (shape[1], shape[0]))
    return img


# ============================================================
# 2. Demo: Point operations
# ============================================================
def demo_point_operations(img_gray):
    print("=" * 50)
    print("[Demo 1] Point operations: Linear stretch / Log / Gamma")
    print("=" * 50)

    img_min, img_max = img_gray.min(), img_gray.max()
    linear = ((img_gray.astype(np.float32) - img_min) /
              (img_max - img_min) * 255).astype(np.uint8)

    c = 255 / np.log(1 + 255)
    log_img = (c * np.log(1 + img_gray.astype(np.float32))).astype(np.uint8)

    gamma = 0.5
    gamma_img = (255 * (img_gray.astype(np.float32) / 255) ** gamma).astype(np.uint8)

    show_images(
        [img_gray, linear, log_img, gamma_img],
        ['Original (dark)', 'Linear stretch', 'Log transform (brighten dark)', f'Gamma (gamma={gamma})'],
        save_path='demo_point_ops.png'
    )


# ============================================================
# 3. Demo: Histogram equalization
# ============================================================
def demo_histogram_equalization(img_gray):
    print("=" * 50)
    print("[Demo 2] Histogram equalization vs CLAHE (adaptive)")
    print("=" * 50)

    eq = cv2.equalizeHist(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_gray)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    imgs = [img_gray, eq, clahe_img]
    names = ['Original', 'Global Eq', 'CLAHE']
    for col, (im, name) in enumerate(zip(imgs, names)):
        axes[0, col].imshow(im, cmap='gray')
        axes[0, col].set_title(name)
        axes[0, col].axis('off')
        axes[1, col].hist(im.ravel(), bins=256, range=(0, 256), color='steelblue')
        axes[1, col].set_title(f'{name} Histogram')
        axes[1, col].set_xlim([0, 256])
    plt.tight_layout()
    plt.savefig('demo_equalization.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_equalization.png")
    plt.close()
    return clahe_img


# ============================================================
# 4. Demo: Affine transform (rotation correction)
# ============================================================
def demo_affine_transform(img_gray):
    print("=" * 50)
    print("[Demo 3] Affine transform: Rotate / Translate / Scale")
    print("=" * 50)

    h, w = img_gray.shape
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, -8, 1.0)
    rotated = cv2.warpAffine(img_gray, M_rot, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

    M_trans = np.float32([[1, 0, 20], [0, 1, 10]])
    translated = cv2.warpAffine(img_gray, M_trans, (w, h))

    resized = cv2.resize(img_gray, (300, 100), interpolation=cv2.INTER_LINEAR)

    show_images(
        [img_gray, rotated, translated, resized],
        ['Original (tilted)', 'Rotation correction (-8 deg)', 'Translation (+20,+10)', 'Resize to 300x100'],
        figsize=(16, 4),
        save_path='demo_affine.png'
    )
    return rotated


# ============================================================
# 5. Demo: Perspective transform (trapezoid -> rectangle)
# ============================================================
def demo_perspective_transform(img_gray):
    print("=" * 50)
    print("[Demo 4] Perspective transform: Simulate and correct trapezoid distortion")
    print("=" * 50)

    h, w = img_gray.shape
    src_pts = np.float32([[20, 10], [w - 20, 10],
                           [w - 5, h - 5], [5, h - 5]])  # trapezoid (simulate oblique shot)
    dst_pts = np.float32([[0, 0], [w, 0],
                           [w, h], [0, h]])  # standard rectangle

    M_forward = cv2.getPerspectiveTransform(dst_pts, src_pts)
    distorted = cv2.warpPerspective(img_gray, M_forward, (w, h))

    M_correct = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(distorted, M_correct, (w, h))

    show_images(
        [img_gray, distorted, corrected],
        ['Original', 'Simulated distortion (trapezoid)', 'Perspective corrected (rectangle)'],
        figsize=(15, 5),
        save_path='demo_perspective.png'
    )
    print(f"\nPerspective transform matrix M:\n{np.round(M_correct, 4)}")
    return corrected


# ============================================================
# 6. Full preprocessing pipeline (license plate scenario)
# ============================================================
def full_preprocessing_pipeline(img_gray):
    print("=" * 50)
    print("[Demo 5] Full preprocessing pipeline (plate scenario)")
    print("=" * 50)
    print("Steps: Original -> CLAHE -> Rotation -> Perspective -> Resize")

    h, w = img_gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    step1 = clahe.apply(img_gray)

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -8, 1.0)
    step2 = cv2.warpAffine(step1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    step3 = cv2.resize(step2, (400, 120), interpolation=cv2.INTER_LINEAR)

    step4_norm = step3.astype(np.float32) / 255.0
    step4_display = (step4_norm * 255).astype(np.uint8)

    show_images(
        [img_gray, step1, step2, step3, step4_display],
        ['Raw input', 'Step1 CLAHE', 'Step2 Rotation', 'Step3 Resize', 'Step4 Normalized [0,1]'],
        figsize=(20, 4),
        save_path='demo_pipeline.png'
    )
    print(f"\nOutput size: {step3.shape}, value range: [{step4_norm.min():.2f}, {step4_norm.max():.2f}]")
    return step3


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    test_img_path = 'test_plate.jpg'
    if os.path.exists(test_img_path):
        img_gray = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded: {test_img_path}, size: {img_gray.shape}")
    else:
        img_gray = create_fake_plate()
        print("No test image found - generated synthetic plate")

    demo_point_operations(img_gray)
    enhanced = demo_histogram_equalization(img_gray)
    rotated = demo_affine_transform(img_gray)
    corrected = demo_perspective_transform(img_gray)
    result = full_preprocessing_pipeline(img_gray)
    print("\n[OK] Stage 1 preprocessing demo complete!")
