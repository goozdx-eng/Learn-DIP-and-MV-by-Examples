"""
第2阶段：图像增强与复原 OpenCV 实例
涵盖：空域滤波（均值/高斯/中值/双边）、频域滤波（FFT低通/高通）、维纳滤波、Canny边缘检测

运行环境：pip install opencv-python numpy matplotlib scipy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 工具：添加不同类型噪声
# ============================================================
def add_gaussian_noise(img, mean=0, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(img, prob=0.05):
    """添加椒盐噪声"""
    noisy = img.copy()
    # 椒（黑点）
    mask = np.random.random(img.shape) < prob / 2
    noisy[mask] = 0
    # 盐（白点）
    mask = np.random.random(img.shape) < prob / 2
    noisy[mask] = 255
    return noisy


def add_motion_blur(img, kernel_size=15, angle=0):
    """添加运动模糊（模拟车辆运动）"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    # 旋转核以模拟不同方向运动
    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred, kernel


def show_comparison(imgs, titles, figsize=(18, 4)):
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# 生成测试图像
# ============================================================
def create_test_image():
    """创建带文字的测试图像模拟车牌"""
    img = np.zeros((120, 400), dtype=np.uint8)
    img[:] = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'A1B234C', (20, 85), font, 1.8, 220, 3)
    return img


# ============================================================
# 演示1：空域滤波对比（去噪）
# ============================================================
def demo_spatial_filtering():
    print("=" * 50)
    print("【演示1】空域滤波：不同滤波器对不同噪声的效果对比")
    print("=" * 50)

    img = create_test_image()
    img_gauss_noisy = add_gaussian_noise(img, sigma=30)
    img_sp_noisy = add_salt_pepper_noise(img, prob=0.08)

    # ---- 高斯噪声的处理对比 ----
    mean_filtered = cv2.blur(img_gauss_noisy, (5, 5))
    gauss_filtered = cv2.GaussianBlur(img_gauss_noisy, (5, 5), 0)
    median_filtered = cv2.medianBlur(img_gauss_noisy, 5)
    bilateral_filtered = cv2.bilateralFilter(img_gauss_noisy, 9, 75, 75)

    show_comparison(
        [img_gauss_noisy, mean_filtered, gauss_filtered, median_filtered, bilateral_filtered],
        ['含高斯噪声', '均值滤波(5×5)', '高斯滤波(5×5)', '中值滤波(5×5)', '双边滤波']
    )

    # ---- 椒盐噪声的处理对比 ----
    mean_sp = cv2.blur(img_sp_noisy, (5, 5))
    gauss_sp = cv2.GaussianBlur(img_sp_noisy, (5, 5), 0)
    median_sp = cv2.medianBlur(img_sp_noisy, 5)

    show_comparison(
        [img_sp_noisy, mean_sp, gauss_sp, median_sp],
        ['含椒盐噪声', '均值滤波（效果差）', '高斯滤波（效果差）', '中值滤波（最佳）']
    )
    print("结论：椒盐噪声 → 必须用中值滤波！均值和高斯会把噪点扩散")


# ============================================================
# 演示2：锐化滤波（增强边缘）
# ============================================================
def demo_sharpening():
    print("=" * 50)
    print("【演示2】锐化滤波：拉普拉斯增强边缘细节")
    print("=" * 50)

    img = create_test_image()
    # 先轻微模糊模拟图像不清晰
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

    # 拉普拉斯二阶微分
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap_abs = np.uint8(np.absolute(lap))

    # 锐化：原图 - 拉普拉斯（注意符号）
    sharpened = cv2.addWeighted(blurred, 1.5, lap_abs, -0.5, 0)

    # Unsharp Masking（反锐化遮蔽）：更温和的锐化
    blurred2 = cv2.GaussianBlur(blurred, (0, 0), 2.0)
    unsharp = cv2.addWeighted(blurred, 1.5, blurred2, -0.5, 0)

    show_comparison(
        [img, blurred, sharpened, unsharp],
        ['原图', '模糊版（待处理）', '拉普拉斯锐化', '反锐化遮蔽(USM)']
    )


# ============================================================
# 演示3：频域滤波（FFT）
# ============================================================
def demo_frequency_filtering():
    print("=" * 50)
    print("【演示3】频域滤波：FFT 低通/高通滤波器")
    print("=" * 50)

    img = create_test_image().astype(np.float32)
    h, w = img.shape

    # 计算FFT并移频到中心
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)  # 频谱可视化

    # ---- 高斯低通滤波器 ----
    sigma_lp = 30
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
    H_lp = np.exp(-(x**2 + y**2) / (2 * sigma_lp**2))

    # ---- 高通滤波器（1 - 低通）----
    H_hp = 1 - H_lp

    # 应用滤波器
    lp_result = np.fft.ifft2(np.fft.ifftshift(fshift * H_lp)).real
    hp_result = np.fft.ifft2(np.fft.ifftshift(fshift * H_hp)).real

    lp_display = np.clip(lp_result, 0, 255).astype(np.uint8)
    hp_display = np.clip(hp_result + 128, 0, 255).astype(np.uint8)  # 加128偏置便于显示

    # 显示频谱
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('原图')
    axes[0, 1].imshow(magnitude, cmap='hot'); axes[0, 1].set_title('频谱（对数幅度）')
    axes[0, 2].imshow(H_lp, cmap='gray'); axes[0, 2].set_title('高斯低通滤波器')
    axes[1, 0].imshow(lp_display, cmap='gray'); axes[1, 0].set_title('低通结果（平滑）')
    axes[1, 1].imshow(hp_display, cmap='gray'); axes[1, 1].set_title('高通结果（边缘）')
    axes[1, 2].imshow(H_hp, cmap='gray'); axes[1, 2].set_title('高通滤波器')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"低通结果范围：[{lp_result.min():.1f}, {lp_result.max():.1f}]")


# ============================================================
# 演示4：维纳滤波复原运动模糊
# ============================================================
def demo_wiener_restoration():
    print("=" * 50)
    print("【演示4】维纳滤波：运动模糊图像的复原")
    print("=" * 50)

    img = create_test_image().astype(np.float32)

    # 添加运动模糊
    blurred, psf = add_motion_blur(img.astype(np.uint8), kernel_size=20, angle=0)
    blurred_noisy = add_gaussian_noise(blurred, sigma=5)

    # scipy维纳滤波
    restored = wiener(blurred_noisy.astype(np.float64), mysize=15, noise=0.01)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    # 频域维纳滤波（手动实现，更能体现原理）
    def wiener_filter_freq(blurred, psf, K=0.01):
        """频域维纳滤波"""
        h, w = blurred.shape
        psf_pad = np.zeros_like(blurred, dtype=np.float32)
        ph, pw = psf.shape
        psf_pad[:ph, :pw] = psf
        PSF = np.fft.fft2(psf_pad)
        G = np.fft.fft2(blurred.astype(np.float32))
        # H* / (|H|^2 + K)
        H_conj = np.conj(PSF)
        W = H_conj / (np.abs(PSF)**2 + K)
        F_hat = W * G
        f_hat = np.abs(np.fft.ifft2(F_hat))
        return np.clip(f_hat, 0, 255).astype(np.uint8)

    restored_freq = wiener_filter_freq(blurred_noisy, psf, K=0.005)

    show_comparison(
        [img.astype(np.uint8), blurred, blurred_noisy, restored, restored_freq],
        ['原图', '运动模糊', '模糊+噪声', 'scipy维纳滤波', '频域维纳滤波(K=0.005)']
    )


# ============================================================
# 演示5：Canny边缘检测（车牌字符边缘提取）
# ============================================================
def demo_canny_edge():
    print("=" * 50)
    print("【演示5】Canny边缘检测：最优边缘算子")
    print("=" * 50)

    img = create_test_image()
    # 先去噪
    img_denoised = cv2.GaussianBlur(img, (3, 3), 0)

    # 不同算子对比
    sobel_x = cv2.Sobel(img_denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)

    laplacian = np.abs(cv2.Laplacian(img_denoised, cv2.CV_64F)).astype(np.uint8)

    # Canny：低阈值50，高阈值150（比例1:3）
    canny_loose = cv2.Canny(img_denoised, 30, 90)    # 宽松阈值
    canny_strict = cv2.Canny(img_denoised, 50, 150)  # 严格阈值（推荐）
    canny_tight = cv2.Canny(img_denoised, 80, 240)   # 过严（漏边缘）

    show_comparison(
        [img, sobel, laplacian, canny_loose, canny_strict, canny_tight],
        ['原图', 'Sobel', 'Laplacian', 'Canny(30,90)', 'Canny(50,150)推荐', 'Canny(80,240)过严'],
        figsize=(20, 4)
    )

    print("\nCanny三步：高斯平滑 → 梯度计算 → 非极大值抑制 → 双阈值连接")
    print("经验：高低阈值比 = 3:1 或 2:1，高阈值 = Otsu阈值 × 0.7")

    # 自动确定Canny阈值（基于Otsu）
    _, otsu_thresh = cv2.threshold(img_denoised, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    auto_high = otsu_thresh
    auto_low = otsu_thresh * 0.5
    canny_auto = cv2.Canny(img_denoised, auto_low, auto_high)
    print(f"Otsu自动阈值：low={auto_low:.0f}, high={auto_high:.0f}")
    show_comparison([canny_strict, canny_auto], ['手动阈值(50,150)', f'Otsu自动({auto_low:.0f},{auto_high:.0f})'])


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    demo_spatial_filtering()
    demo_sharpening()
    demo_frequency_filtering()
    demo_wiener_restoration()
    demo_canny_edge()

    print("\n✅ 第2阶段增强与复原演示完成！下一步：进入第3阶段「目标分离」")
