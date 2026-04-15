"""
Stage 3: Object Segmentation - OpenCV Examples
Covers: Binarization (Otsu/adaptive), Morphology (erode/dilate/open/close/top-hat),
        Connected components, Contour detection, Character cutting

Install: pip install opencv-python numpy matplotlib
Run: python demo_segmentation.py
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Generate synthetic license plate (with border, chars, rivet noise)
# ============================================================
def create_license_plate(h=120, w=400):
    img = np.ones((h, w), dtype=np.uint8) * 200
    cv2.rectangle(img, (3, 3), (w-4, h-4), 50, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    chars = ['B', 'A', '1', '2', '3', '4', 'C']
    for i, ch in enumerate(chars):
        x = 20 + i * 54
        cv2.putText(img, ch, (x, 90), font, 1.6, 30, 3)
    for pos in [(15, 60), (385, 60), (15, 30), (385, 30)]:
        cv2.circle(img, pos, 6, 30, -1)
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def show_images(imgs, titles, cmap='gray', figsize=(18, 4), save_path=None):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] -> {save_path}")
    plt.close()


# ============================================================
# Demo 1: Binarization methods comparison
# ============================================================
def demo_binarization(img):
    print("=" * 50)
    print("[Demo 1] Binarization: Fixed / Otsu / Adaptive")
    print("=" * 50)

    _, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    otsu_thresh, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)

    show_images(
        [img, binary_fixed, binary_otsu, binary_adaptive],
        ['Original', f'Fixed T=127', f'Otsu T={otsu_thresh:.0f}', 'Adaptive (block=21)'],
        save_path='demo_binarization.png'
    )
    print(f"Otsu optimal threshold: {otsu_thresh:.0f}")
    return binary_otsu


# ============================================================
# Demo 2: Morphological basics
# ============================================================
def demo_morphology_basics(binary):
    print("=" * 50)
    print("[Demo 2] Morphological ops: Erode, Dilate, Open, Close")
    print("=" * 50)

    se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    se5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded = cv2.erode(binary, se3)
    dilated = cv2.dilate(binary, se3)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se5)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se5)

    show_images(
        [binary, eroded, dilated, opened, closed],
        ['Binary (input)', 'Erode 3x3', 'Dilate 3x3', 'Open 5x5 (removes noise)', 'Close 5x5 (fills holes)'],
        save_path='demo_morph_basic.png'
    )

    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, se5)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, se5)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, se3)

    show_images(
        [tophat, blackhat, gradient],
        ['TopHat (bright small objects)', 'BlackHat (dark small objects)', 'Morphological Gradient (outline)'],
        save_path='demo_morph_advanced.png'
    )


# ============================================================
# Demo 3: Remove rivet noise (open operation实战)
# ============================================================
def demo_remove_rivets(binary):
    print("=" * 50)
    print("[Demo 3] Open operation removes rivets (chars preserved, dots eliminated)")
    print("=" * 50)

    se_rivet = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_rivet)

    num_before, _, stats_before, _ = cv2.connectedComponentsWithStats(binary)
    num_after, _, stats_after, _ = cv2.connectedComponentsWithStats(cleaned)

    show_images([binary, cleaned],
                ['Before open (rivets present)', 'After open (rivets removed)'],
                save_path='demo_rivets.png')
    print(f"Connected components before: {num_before - 1}")
    print(f"Connected components after: {num_after - 1}")
    return cleaned


# ============================================================
# Demo 4: Connected components + character cutting
# ============================================================
def demo_connected_components(img, binary):
    print("=" * 50)
    print("[Demo 4] Connected components: Per-character cutting and filtering")
    print("=" * 50)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

    print(f"Total connected components (incl. background): {num_labels}")
    print(f"{'Label':>5} {'X':>5} {'Y':>5} {'W':>5} {'H':>5} {'Area':>7} {'Keep?':>6}")
    print("-" * 45)

    h, w = img.shape
    char_regions = []
    min_area = 200
    max_area = h * w * 0.15
    min_h = h * 0.3

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i][:5]
        keep = (min_area < area < max_area) and (bh > min_h)
        mark = "[OK]" if keep else "[X]"
        print(f"{i:>5} {x:>5} {y:>5} {bw:>5} {bh:>5} {area:>7} {mark:>6}")
        if keep:
            char_regions.append((x, y, bw, bh, area))

    char_regions.sort(key=lambda r: r[0])

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 128)]
    for idx, (x, y, bw, bh, _) in enumerate(char_regions):
        color = colors[idx % len(colors)]
        cv2.rectangle(img_color, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(img_color, str(idx+1), (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Main figure
    fig, axes = plt.subplots(2, max(1, len(char_regions)), figsize=(16, 5))
    if len(char_regions) == 1:
        axes = axes.reshape(2, 1)

    ax_main = plt.subplot2grid((2, max(1, len(char_regions))), (0, 0),
                                colspan=max(1, len(char_regions)))
    ax_main.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f'Connected component result ({len(char_regions)} chars detected)')
    ax_main.axis('off')

    # Individual chars
    plt.figure(figsize=(16, 3))
    for idx, (x, y, bw, bh, _) in enumerate(char_regions):
        char_img = img[y:y+bh, x:x+bw]
        char_resized = cv2.resize(char_img, (40, 80))
        plt.subplot(1, len(char_regions), idx+1)
        plt.imshow(char_resized, cmap='gray')
        plt.title(f'Char {idx+1}')
        plt.axis('off')
    plt.suptitle('Cut characters (normalized to 40x80)')
    plt.tight_layout()
    plt.savefig('demo_chars.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_chars.png")
    plt.close()
    return char_regions


# ============================================================
# Demo 5: Contour detection
# ============================================================
def demo_contour_detection(img, binary):
    print("=" * 50)
    print("[Demo 5] Contour detection: findContours + moment features")
    print("=" * 50)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)
    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

    print(f"Detected contours: {len(contours)}")
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.intp)
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img_color, (cx, cy), 3, (0, 0, 255), -1)
        print(f"  Contour {i}: area={area:.0f}, center=({cx},{cy}), minRect={rect[1]}")

    plt.figure(figsize=(14, 4))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('Contours (green) + Centroids (red dots)')
    plt.axis('off')
    plt.savefig('demo_contours.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_contours.png")
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    plate = create_license_plate()
    print(f"Plate size: {plate.shape}, range: [{plate.min()}, {plate.max()}]")

    binary = demo_binarization(plate)
    demo_morphology_basics(binary)
    cleaned_binary = demo_remove_rivets(binary)
    char_regions = demo_connected_components(plate, cleaned_binary)
    demo_contour_detection(plate, cleaned_binary)

    print(f"\nCut {len(char_regions)} character regions")
    print("\n[OK] Stage 3 Object Segmentation demo complete!")
