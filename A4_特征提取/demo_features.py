"""
第4阶段：特征提取 OpenCV 实例
涵盖：HOG特征、LBP特征、PCA降维、特征可视化与对比

运行环境：pip install opencv-python numpy matplotlib scikit-learn scikit-image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 生成字符样本集（模拟切割出的车牌字符）
# ============================================================
def generate_char_samples():
    """生成模拟字符图像样本，包含光照变化和轻微噪声"""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    samples = []
    labels = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    np.random.seed(42)
    for char in ['0', '8', 'B', 'D', '1', 'I', 'l']:  # 选几个相似字符
        for i in range(20):  # 每类20个样本
            img = np.ones((80, 40), dtype=np.uint8) * 200
            # 随机光照变化
            brightness = np.random.randint(150, 230)
            img[:] = brightness
            # 随机字体大小微小变化
            scale = 1.5 + np.random.uniform(-0.2, 0.2)
            thickness = np.random.randint(2, 4)
            # 随机位置偏移
            dx = np.random.randint(-3, 3)
            dy = np.random.randint(-3, 3)
            cv2.putText(img, char, (5+dx, 65+dy), font, scale, 30, thickness)
            # 添加高斯噪声
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            samples.append(img)
            labels.append(char)

    return np.array(samples), labels


def show_sample_chars(samples, labels, n_per_class=5):
    """显示样本示例"""
    unique_chars = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(len(unique_chars), n_per_class, figsize=(12, 14))
    for row, char in enumerate(unique_chars):
        idxs = [i for i, l in enumerate(labels) if l == char][:n_per_class]
        for col, idx in enumerate(idxs):
            axes[row, col].imshow(samples[idx], cmap='gray')
            if col == 0:
                axes[row, col].set_ylabel(f'"{char}"', fontsize=12, rotation=0, labelpad=20)
            axes[row, col].axis('off')
    plt.suptitle('字符样本（含光照/噪声变化）', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 演示1：HOG 特征提取
# ============================================================
def compute_hog(img, cell_size=8, block_size=2, nbins=9):
    """手动实现简化版HOG，便于理解原理"""
    h, w = img.shape
    # Step1: 计算梯度
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180  # 无符号角度

    # Step2: Cell直方图
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

    # Step3: Block归一化 + 拼接
    hog_features = []
    for i in range(n_cells_h - block_size + 1):
        for j in range(n_cells_w - block_size + 1):
            block = cell_hists[i:i+block_size, j:j+block_size, :].flatten()
            norm = np.linalg.norm(block) + 1e-6
            hog_features.extend(block / norm)

    return np.array(hog_features), mag, angle, cell_hists


def demo_hog_feature(samples, labels):
    print("=" * 50)
    print("【演示1】HOG 特征提取与可视化")
    print("=" * 50)

    # 取两个典型字符做对比（'8' vs 'B'）
    idx_8 = next(i for i, l in enumerate(labels) if l == '8')
    idx_B = next(i for i, l in enumerate(labels) if l == 'B')

    for char, idx in [('8', idx_8), ('B', idx_B)]:
        img = samples[idx]
        feat, mag, angle, cell_hists = compute_hog(img)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'字符 "{char}" 原图')
        axes[1].imshow(mag, cmap='hot')
        axes[1].set_title('梯度幅度图')
        axes[2].imshow(angle, cmap='hsv')
        axes[2].set_title('梯度方向图')
        # HOG特征向量可视化（前100维）
        axes[3].bar(range(min(100, len(feat))), feat[:100], color='steelblue', width=1)
        axes[3].set_title(f'HOG特征向量(前100维，总{len(feat)}维)')
        for ax in axes[:3]:
            ax.axis('off')
        plt.suptitle(f'字符 "{char}" 的HOG特征')
        plt.tight_layout()
        plt.show()

        print(f'字符 "{char}" HOG特征维度：{len(feat)}')

    # 用OpenCV内置HOG（更高效）
    hog_cv = cv2.HOGDescriptor(
        _winSize=(40, 80), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    feat_8_cv = hog_cv.compute(samples[idx_8]).flatten()
    feat_B_cv = hog_cv.compute(samples[idx_B]).flatten()
    dist = np.linalg.norm(feat_8_cv - feat_B_cv)
    print(f'\nOpenCV HOG特征维度：{len(feat_8_cv)}')
    print(f'"8" 和 "B" 的HOG特征欧氏距离：{dist:.2f}')

    return hog_cv


# ============================================================
# 演示2：LBP 特征提取
# ============================================================
def compute_lbp_manual(img, radius=1, n_points=8):
    """手动实现圆形LBP（便于理解）"""
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
                # 双线性插值
                nx_int, ny_int = int(nx), int(ny)
                if 0 <= nx_int < w-1 and 0 <= ny_int < h-1:
                    neighbor = img[ny_int, nx_int]
                else:
                    neighbor = 0
                if neighbor >= center:
                    binary_code |= (1 << (n_points - 1 - p))
            lbp_img[y, x] = binary_code % 256

    return lbp_img


def demo_lbp_feature(samples, labels):
    print("=" * 50)
    print("【演示2】LBP 特征：纹理描述，对光照不变")
    print("=" * 50)

    try:
        from skimage.feature import local_binary_pattern
        use_skimage = True
    except ImportError:
        use_skimage = False
        print("skimage未安装，使用手动LBP实现（较慢）")

    idx_8 = next(i for i, l in enumerate(labels) if l == '8')
    idx_B = next(i for i, l in enumerate(labels) if l == 'B')

    results = {}
    for char, idx in [('8', idx_8), ('B', idx_B)]:
        img = samples[idx]
        if use_skimage:
            lbp_img = local_binary_pattern(img, P=8, R=1, method='uniform')
            lbp_img = lbp_img.astype(np.uint8)
        else:
            lbp_img = compute_lbp_manual(img)

        # 统计LBP直方图（特征向量）
        hist, _ = np.histogram(lbp_img.ravel(), bins=59, range=(0, 59))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6  # 归一化

        results[char] = {'lbp': lbp_img, 'hist': hist}

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'字符"{char}"原图')
        axes[1].imshow(lbp_img, cmap='gray')
        axes[1].set_title('LBP编码图')
        axes[2].bar(range(len(hist)), hist, color='coral', width=0.8)
        axes[2].set_title('均匀LBP直方图（58+1维特征）')
        axes[2].set_xlabel('LBP编码')
        axes[2].set_ylabel('归一化频率')
        axes[0].axis('off')
        axes[1].axis('off')
        plt.suptitle(f'字符"{char}"的LBP特征')
        plt.tight_layout()
        plt.show()

    # 验证光照不变性
    print("\n验证LBP的光照不变性：")
    idx_bright = next(i for i, l in enumerate(labels) if l == '0')
    img_orig = samples[idx_bright]
    img_dark = np.clip(img_orig.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    img_bright = np.clip(img_orig.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

    def get_lbp_hist(im):
        if use_skimage:
            lbp = local_binary_pattern(im, P=8, R=1, method='uniform')
        else:
            lbp = compute_lbp_manual(im)
        h, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        h = h.astype(np.float32)
        return h / (h.sum() + 1e-6)

    h_orig = get_lbp_hist(img_orig)
    h_dark = get_lbp_hist(img_dark)
    h_bright = get_lbp_hist(img_bright)

    dist_dark = np.linalg.norm(h_orig - h_dark)
    dist_bright = np.linalg.norm(h_orig - h_bright)
    print(f"原图 vs 暗图（×0.5）LBP直方图距离：{dist_dark:.4f}")
    print(f"原图 vs 亮图（×1.5）LBP直方图距离：{dist_bright:.4f}")
    print("距离越小说明LBP对光照变化越不敏感")


# ============================================================
# 演示3：PCA 降维与可视化
# ============================================================
def demo_pca(samples, labels):
    print("=" * 50)
    print("【演示3】PCA 降维：Eigenfaces思想用于字符识别")
    print("=" * 50)

    # 展平为向量
    X = samples.reshape(len(samples), -1).astype(np.float32)
    y = np.array(labels)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # 方差贡献率
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"原始维度：{X.shape[1]}")
    print(f"达到95%累积方差所需主成分数：{n_95}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(cumvar, 'b-o', markersize=3)
    axes[0].axhline(0.95, color='r', linestyle='--', label='95%方差')
    axes[0].axvline(n_95, color='g', linestyle='--', label=f'PC={n_95}')
    axes[0].set_xlabel('主成分数')
    axes[0].set_ylabel('累积方差贡献率')
    axes[0].set_title('PCA方差解释率曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2D散点图
    colors_map = {'0':'red', '8':'blue', 'B':'green', 'D':'orange',
                  '1':'purple', 'I':'brown', 'l':'pink'}
    for char in set(labels):
        mask = y == char
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors_map.get(char, 'gray'),
                        label=f'"{char}"', alpha=0.7, s=50)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA 2D投影（前2个主成分）')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 可视化主成分（Eigenchar）
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.ravel()):
        if i < 10:
            eigenchar = pca.components_[i].reshape(80, 40)
            ax.imshow(eigenchar, cmap='RdBu_r')
            ax.set_title(f'PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.axis('off')
    plt.suptitle('前10个主成分（Eigenchar）—— 可视化主要变化方向')
    plt.tight_layout()
    plt.show()

    print(f"\nPCA后前2维能区分 '0' vs '8' vs 'B' 等相似字符")

    return pca, scaler


# ============================================================
# 演示4：特征对比（用距离衡量可分性）
# ============================================================
def demo_feature_comparison(samples, labels):
    print("=" * 50)
    print("【演示4】特征可分性对比：原始像素 vs HOG vs LBP vs PCA")
    print("=" * 50)

    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        print("需要安装scikit-image：pip install scikit-image")
        return

    hog_cv = cv2.HOGDescriptor(
        _winSize=(40, 80), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    # 提取各类特征
    pixel_feats, hog_feats, lbp_feats = [], [], []
    for img in samples:
        pixel_feats.append(img.flatten().astype(np.float32) / 255.0)
        hog_feats.append(hog_cv.compute(img).flatten())
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        lbp_feats.append(hist)

    # PCA特征
    X = np.array(pixel_feats)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=30)
    pca_feats = pca.fit_transform(X_scaled)

    # 计算类内/类间距离（衡量可分性）
    def compute_separability(feats, labels_arr):
        unique = list(set(labels_arr))
        intra_dists, inter_dists = [], []
        feats_arr = np.array(feats)
        for c in unique:
            mask = labels_arr == c
            class_feats = feats_arr[mask]
            # 类内距离
            if len(class_feats) > 1:
                for i in range(len(class_feats)):
                    for j in range(i+1, len(class_feats)):
                        intra_dists.append(np.linalg.norm(class_feats[i] - class_feats[j]))
            # 类间距离
            other_mask = labels_arr != c
            other_feats = feats_arr[other_mask]
            for cf in class_feats:
                for of in other_feats[:5]:  # 只取前5个加速
                    inter_dists.append(np.linalg.norm(cf - of))
        return np.mean(intra_dists), np.mean(inter_dists)

    labels_arr = np.array(labels)
    results = {}
    for name, feats in [('原始像素', pixel_feats), ('HOG', hog_feats),
                         ('LBP', lbp_feats), ('PCA(30维)', list(pca_feats))]:
        intra, inter = compute_separability(feats, labels_arr)
        ratio = inter / (intra + 1e-6)
        results[name] = {'类内距离': intra, '类间距离': inter, '可分性比': ratio}
        print(f"{name:12s}：维度={len(feats[0]):4d}，类内距={intra:.3f}，类间距={inter:.3f}，可分性比={ratio:.2f}")

    # 可视化
    names = list(results.keys())
    ratios = [results[n]['可分性比'] for n in names]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, ratios, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    plt.ylabel('类间距/类内距（越大越好）')
    plt.title('不同特征的可分性对比（比值越高识别越容易）')
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("生成字符样本集...")
    samples, labels = generate_char_samples()
    print(f"样本数：{len(samples)}，字符类别：{set(labels)}")
    print(f"单张图像尺寸：{samples[0].shape}")

    show_sample_chars(samples, labels, n_per_class=4)
    hog_cv = demo_hog_feature(samples, labels)
    demo_lbp_feature(samples, labels)
    pca, scaler = demo_pca(samples, labels)
    demo_feature_comparison(samples, labels)

    print("\n✅ 第4阶段特征提取演示完成！下一步：进入第5阶段「分类识别」")
