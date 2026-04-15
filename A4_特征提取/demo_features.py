"""
Stage 4: Feature Extraction - OpenCV Examples
Covers: HOG features, LBP features, PCA dimensionality reduction, Feature comparison

Install: pip install opencv-python numpy matplotlib scikit-learn scikit-image
Run: python demo_features.py
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[INFO] scikit-image not installed, using manual LBP (slower but complete)")


# ============================================================
# Generate character sample set
# ============================================================
def generate_char_samples():
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    samples, labels = [], []
    font = cv2.FONT_HERSHEY_SIMPLEX

    np.random.seed(42)
    for char in ['0', '8', 'B', 'D', '1', 'I', 'l']:
        for i in range(20):
            img = np.ones((80, 40), dtype=np.uint8) * 200
            brightness = np.random.randint(150, 230)
            img[:] = brightness
            scale = 1.5 + np.random.uniform(-0.2, 0.2)
            thickness = np.random.randint(2, 4)
            dx = np.random.randint(-3, 3)
            dy = np.random.randint(-3, 3)
            cv2.putText(img, char, (5+dx, 65+dy), font, scale, 30, thickness)
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            samples.append(img)
            labels.append(char)
    return np.array(samples), labels


def show_sample_chars(samples, labels, n_per_class=4):
    unique_chars = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(len(unique_chars), n_per_class, figsize=(12, 14))
    for row, char in enumerate(unique_chars):
        idxs = [i for i, l in enumerate(labels) if l == char][:n_per_class]
        for col, idx in enumerate(idxs):
            axes[row, col].imshow(samples[idx], cmap='gray')
            if col == 0:
                axes[row, col].set_ylabel(f'"{char}"', fontsize=12, rotation=0, labelpad=20)
            axes[row, col].axis('off')
    plt.suptitle('Character samples (with illumination/noise variation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_samples.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_samples.png")
    plt.close()


# ============================================================
# Manual HOG implementation
# ============================================================
def compute_hog(img, cell_size=8, block_size=2, nbins=9):
    h, w = img.shape
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180

    n_cells_h = h // cell_size
    n_cells_w = w // cell_size
    cell_hists = np.zeros((n_cells_h, n_cells_w, nbins))
    bin_size = 180 / nbins

    for i in range(n_cells_h):
        for j in range(n_cells_w):
            cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            for bin_idx in range(nbins):
                bin_min = bin_idx * bin_size
                bin_max = (bin_idx + 1) * bin_size
                mask = (cell_ang >= bin_min) & (cell_ang < bin_max)
                cell_hists[i, j, bin_idx] = cell_mag[mask].sum()

    hog_features = []
    for i in range(n_cells_h - block_size + 1):
        for j in range(n_cells_w - block_size + 1):
            block = cell_hists[i:i+block_size, j:j+block_size, :].flatten()
            norm = np.linalg.norm(block) + 1e-6
            hog_features.extend(block / norm)
    return np.array(hog_features), mag, angle, cell_hists


# ============================================================
# Demo 1: HOG feature extraction
# ============================================================
def demo_hog_feature(samples, labels):
    print("=" * 50)
    print("[Demo 1] HOG Feature Extraction & Visualization")
    print("=" * 50)

    idx_8 = next(i for i, l in enumerate(labels) if l == '8')
    idx_B = next(i for i, l in enumerate(labels) if l == 'B')

    for char, idx in [('8', idx_8), ('B', idx_B)]:
        img = samples[idx]
        feat, mag, angle, cell_hists = compute_hog(img)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Char "{char}" original')
        axes[1].imshow(mag, cmap='hot')
        axes[1].set_title('Gradient magnitude')
        axes[2].imshow(angle, cmap='hsv')
        axes[2].set_title('Gradient direction')
        axes[3].bar(range(min(100, len(feat))), feat[:100], color='steelblue', width=1)
        axes[3].set_title(f'HOG vector (first 100 of {len(feat)} dims)')
        for ax in axes[:3]:
            ax.axis('off')
        plt.suptitle(f'HOG features of character "{char}"')
        plt.tight_layout()
        plt.savefig(f'demo_hog_{char}.png', dpi=150, bbox_inches='tight')
        print(f"[Saved] -> demo_hog_{char}.png")
        plt.close()
        print(f'Char "{char}" HOG dimension: {len(feat)}')

    hog_cv = cv2.HOGDescriptor(
        _winSize=(40, 80), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    feat_8_cv = hog_cv.compute(samples[idx_8]).flatten()
    feat_B_cv = hog_cv.compute(samples[idx_B]).flatten()
    dist = np.linalg.norm(feat_8_cv - feat_B_cv)
    print(f'\nOpenCV HOG dimension: {len(feat_8_cv)}')
    print(f'"8" vs "B" HOG Euclidean distance: {dist:.2f}')
    return hog_cv


# ============================================================
# Manual LBP implementation
# ============================================================
def compute_lbp_manual(img, radius=1, n_points=8):
    h, w = img.shape
    lbp_img = np.zeros_like(img, dtype=np.uint8)
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = img[y, x]
            binary_code = 0
            for p in range(n_points):
                angle = 2 * np.pi * p / n_points
                nx = x + radius * np.cos(angle)
                ny = y - radius * np.sin(angle)
                nx_int, ny_int = int(nx), int(ny)
                if 0 <= nx_int < w-1 and 0 <= ny_int < h-1:
                    neighbor = img[ny_int, nx_int]
                else:
                    neighbor = 0
                if neighbor >= center:
                    binary_code |= (1 << (n_points - 1 - p))
            lbp_img[y, x] = binary_code % 256
    return lbp_img


# ============================================================
# Demo 2: LBP feature
# ============================================================
def demo_lbp_feature(samples, labels):
    print("=" * 50)
    print("[Demo 2] LBP Feature: Texture descriptor, illumination-invariant")
    print("=" * 50)

    idx_8 = next(i for i, l in enumerate(labels) if l == '8')
    idx_B = next(i for i, l in enumerate(labels) if l == 'B')

    results = {}
    for char, idx in [('8', idx_8), ('B', idx_B)]:
        img = samples[idx]
        if HAS_SKIMAGE:
            lbp_img = local_binary_pattern(img, P=8, R=1, method='uniform')
            lbp_img = lbp_img.astype(np.uint8)
        else:
            lbp_img = compute_lbp_manual(img)

        hist, _ = np.histogram(lbp_img.ravel(), bins=59, range=(0, 59))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        results[char] = {'lbp': lbp_img, 'hist': hist}

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Char "{char}" original')
        axes[1].imshow(lbp_img, cmap='gray')
        axes[1].set_title('LBP coded image')
        axes[2].bar(range(len(hist)), hist, color='coral', width=0.8)
        axes[2].set_title('Uniform LBP histogram (58+1 dims)')
        axes[2].set_xlabel('LBP code')
        axes[2].set_ylabel('Normalized frequency')
        axes[0].axis('off')
        axes[1].axis('off')
        plt.suptitle(f'LBP features of "{char}"')
        plt.tight_layout()
        plt.savefig(f'demo_lbp_{char}.png', dpi=150, bbox_inches='tight')
        print(f"[Saved] -> demo_lbp_{char}.png")
        plt.close()

    # Verify illumination invariance
    print("\nVerifying LBP illumination invariance:")
    idx_bright = next(i for i, l in enumerate(labels) if l == '0')
    img_orig = samples[idx_bright]
    img_dark = np.clip(img_orig.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    img_bright = np.clip(img_orig.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

    def get_lbp_hist(im):
        if HAS_SKIMAGE:
            lbp = local_binary_pattern(im, P=8, R=1, method='uniform')
        else:
            lbp = compute_lbp_hist(im)
        h, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        h = h.astype(np.float32)
        return h / (h.sum() + 1e-6)

    h_orig = get_lbp_hist(img_orig)
    h_dark = get_lbp_hist(img_dark)
    h_bright = get_lbp_hist(img_bright)

    dist_dark = np.linalg.norm(h_orig - h_dark)
    dist_bright = np.linalg.norm(h_orig - h_bright)
    print(f"Original vs Dark(x0.5) LBP histogram distance: {dist_dark:.4f}")
    print(f"Original vs Bright(x1.5) LBP histogram distance: {dist_bright:.4f}")
    print("Smaller distance = LBP more robust to illumination changes")


# ============================================================
# Demo 3: PCA dimensionality reduction
# ============================================================
def demo_pca(samples, labels):
    print("=" * 50)
    print("[Demo 3] PCA: Eigenfaces idea applied to character recognition")
    print("=" * 50)

    X = samples.reshape(len(samples), -1).astype(np.float32)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"Original dimension: {X.shape[1]}")
    print(f"Components needed for 95% variance: {n_95}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(cumvar, 'b-o', markersize=3)
    axes[0].axhline(0.95, color='r', linestyle='--', label='95% variance')
    axes[0].axvline(n_95, color='g', linestyle='--', label=f'PC={n_95}')
    axes[0].set_xlabel('Number of principal components')
    axes[0].set_ylabel('Cumulative explained variance ratio')
    axes[0].set_title('PCA Variance Explained Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors_map = {'0':'red', '8':'blue', 'B':'green', 'D':'orange',
                  '1':'purple', 'I':'brown', 'l':'pink'}
    for char in set(labels):
        mask = y == char
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors_map.get(char, 'gray'),
                        label=f'"{char}"', alpha=0.7, s=50)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA 2D projection (first 2 principal components)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('demo_pca.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_pca.png")
    plt.close()

    # Eigenchars
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.ravel()):
        if i < 10:
            eigenchar = pca.components_[i].reshape(80, 40)
            ax.imshow(eigenchar, cmap='RdBu_r')
            ax.set_title(f'PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.axis('off')
    plt.suptitle('Top 10 principal components (Eigenchars) - visualizing main variation directions')
    plt.tight_layout()
    plt.savefig('demo_eigenchars.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_eigenchars.png")
    plt.close()

    print(f"\nPCA first 2 dims can distinguish '0' vs '8' vs 'B' etc.")
    return pca, scaler


# ============================================================
# Demo 4: Feature separability comparison
# ============================================================
def demo_feature_comparison(samples, labels):
    print("=" * 50)
    print("[Demo 4] Feature separability: Raw pixels vs HOG vs LBP vs PCA")
    print("=" * 50)

    if not HAS_SKIMAGE:
        print("[WARNING] scikit-image not installed, skipping feature comparison")
        return

    hog_cv = cv2.HOGDescriptor(
        _winSize=(40, 80), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    pixel_feats, hog_feats, lbp_feats = [], [], []
    for img in samples:
        pixel_feats.append(img.flatten().astype(np.float32) / 255.0)
        hog_feats.append(hog_cv.compute(img).flatten())
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        lbp_feats.append(hist)

    X = np.array(pixel_feats)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=30)
    pca_feats = pca.fit_transform(X_scaled)

    def compute_separability(feats, labels_arr):
        unique = list(set(labels_arr))
        intra_dists, inter_dists = [], []
        feats_arr = np.array(feats)
        for c in unique:
            mask = labels_arr == c
            class_feats = feats_arr[mask]
            if len(class_feats) > 1:
                for i in range(len(class_feats)):
                    for j in range(i+1, len(class_feats)):
                        intra_dists.append(np.linalg.norm(class_feats[i] - class_feats[j]))
            other_mask = labels_arr != c
            other_feats = feats_arr[other_mask]
            for cf in class_feats:
                for of in other_feats[:5]:
                    inter_dists.append(np.linalg.norm(cf - of))
        return np.mean(intra_dists), np.mean(inter_dists)

    labels_arr = np.array(labels)
    results = {}
    for name, feats in [('Raw pixels', pixel_feats), ('HOG', hog_feats),
                         ('LBP', lbp_feats), ('PCA(30 dims)', list(pca_feats))]:
        intra, inter = compute_separability(feats, labels_arr)
        ratio = inter / (intra + 1e-6)
        results[name] = {'intra': intra, 'inter': inter, 'ratio': ratio}
        print(f"{name:12s}: dim={len(feats[0]):4d}, intra={intra:.3f}, inter={inter:.3f}, ratio={ratio:.2f}")

    names = list(results.keys())
    ratios = [results[n]['ratio'] for n in names]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, ratios, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    plt.ylabel('Inter-class / Intra-class distance (higher=better)')
    plt.title('Feature separability comparison (higher ratio = easier to classify)')
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_separability.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_separability.png")
    plt.close()


# ============================================================
# Manual LBP for get_lbp_hist helper
# ============================================================
def compute_lbp_hist(img, radius=1, n_points=8):
    """Compute LBP histogram without skimage (for demo_lbp_feature helper)"""
    h, w = img.shape
    lbp = np.zeros_like(img, dtype=np.uint8)
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = img[y, x]
            binary_code = 0
            for p in range(n_points):
                angle = 2 * np.pi * p / n_points
                nx = x + radius * np.cos(angle)
                ny = y - radius * np.sin(angle)
                nx_int, ny_int = int(nx), int(ny)
                if 0 <= nx_int < w-1 and 0 <= ny_int < h-1:
                    neighbor = img[ny_int, nx_int]
                else:
                    neighbor = 0
                if neighbor >= center:
                    binary_code |= (1 << (n_points - 1 - p))
            lbp[y, x] = binary_code % 256
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    return hist.astype(np.float32)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating character sample set...")
    samples, labels = generate_char_samples()
    print(f"Samples: {len(samples)}, chars: {set(labels)}, size: {samples[0].shape}")

    show_sample_chars(samples, labels, n_per_class=4)
    hog_cv = demo_hog_feature(samples, labels)
    demo_lbp_feature(samples, labels)
    pca, scaler = demo_pca(samples, labels)
    demo_feature_comparison(samples, labels)
    print("\n[OK] Stage 4 Feature Extraction demo complete!")
