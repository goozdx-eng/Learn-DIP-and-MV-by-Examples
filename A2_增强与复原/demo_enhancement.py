"""
Stage 2: Image Enhancement & Restoration - OpenCV Examples
Covers: Spatial filtering (mean/Gaussian/median/bilateral), FFT filtering, Wiener filter, Canny edge

Install: pip install opencv-python numpy matplotlib scipy
Run: python demo_enhancement.py
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener


# ============================================================
# Noise generators
# ============================================================
def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy()
    mask = np.random.random(img.shape) < prob / 2
    noisy[mask] = 0
    mask = np.random.random(img.shape) < prob / 2
    noisy[mask] = 255
    return noisy


def add_motion_blur(img, kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred, kernel


def show_comparison(imgs, titles, figsize=(18, 4), save_path=None):
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] -> {save_path}")
    plt.close()


# ============================================================
# Create test image
# ============================================================
def create_test_image():
    img = np.zeros((120, 400), dtype=np.uint8)
    img[:] = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'A1B234C', (20, 85), font, 1.8, 220, 3)
    return img


# ============================================================
# Demo 1: Spatial filtering comparison
# ============================================================
def demo_spatial_filtering():
    print("=" * 50)
    print("[Demo 1] Spatial filtering: Filter comparison on different noise types")
    print("=" * 50)

    img = create_test_image()
    img_gauss = add_gaussian_noise(img, sigma=30)
    img_sp = add_salt_pepper_noise(img, prob=0.08)

    mean_f = cv2.blur(img_gauss, (5, 5))
    gauss_f = cv2.GaussianBlur(img_gauss, (5, 5), 0)
    median_f = cv2.medianBlur(img_gauss, 5)
    bilateral_f = cv2.bilateralFilter(img_gauss, 9, 75, 75)

    show_comparison(
        [img_gauss, mean_f, gauss_f, median_f, bilateral_f],
        ['Gaussian noise', 'Mean 5x5', 'Gaussian 5x5', 'Median 5x5', 'Bilateral filter'],
        save_path='demo_spatial_gauss.png'
    )

    mean_sp = cv2.blur(img_sp, (5, 5))
    gauss_sp = cv2.GaussianBlur(img_sp, (5, 5), 0)
    median_sp = cv2.medianBlur(img_sp, 5)

    show_comparison(
        [img_sp, mean_sp, gauss_sp, median_sp],
        ['Salt-Pepper noise', 'Mean (poor)', 'Gaussian (poor)', 'Median (best for impulse)'],
        save_path='demo_spatial_sp.png'
    )
    print("Conclusion: Salt-pepper -> MUST use median filter!")


# ============================================================
# Demo 2: Sharpening filters
# ============================================================
def demo_sharpening():
    print("=" * 50)
    print("[Demo 2] Sharpening: Laplacian edge enhancement")
    print("=" * 50)

    img = create_test_image()
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap_abs = np.uint8(np.absolute(lap))
    sharpened = cv2.addWeighted(blurred, 1.5, lap_abs, -0.5, 0)
    blurred2 = cv2.GaussianBlur(blurred, (0, 0), 2.0)
    unsharp = cv2.addWeighted(blurred, 1.5, blurred2, -0.5, 0)

    show_comparison(
        [img, blurred, sharpened, unsharp],
        ['Original', 'Blurred (simulated)', 'Laplacian sharpen', 'Unsharp Mask (USM)'],
        save_path='demo_sharpening.png'
    )


# ============================================================
# Demo 3: Frequency domain filtering (FFT)
# ============================================================
def demo_frequency_filtering():
    print("=" * 50)
    print("[Demo 3] Frequency domain: FFT Low-pass / High-pass filters")
    print("=" * 50)

    img = create_test_image().astype(np.float32)
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    sigma_lp = 30
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
    H_lp = np.exp(-(x**2 + y**2) / (2 * sigma_lp**2))
    H_hp = 1 - H_lp

    lp_result = np.fft.ifft2(np.fft.ifftshift(fshift * H_lp)).real
    hp_result = np.fft.ifft2(np.fft.ifftshift(fshift * H_hp)).real
    lp_display = np.clip(lp_result, 0, 255).astype(np.uint8)
    hp_display = np.clip(hp_result + 128, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('Original')
    axes[0, 1].imshow(magnitude, cmap='hot'); axes[0, 1].set_title('Spectrum (log magnitude)')
    axes[0, 2].imshow(H_lp, cmap='gray'); axes[0, 2].set_title('Gaussian Low-pass filter')
    axes[1, 0].imshow(lp_display, cmap='gray'); axes[1, 0].set_title('Low-pass result (smooth)')
    axes[1, 1].imshow(hp_display, cmap='gray'); axes[1, 1].set_title('High-pass result (edges)')
    axes[1, 2].imshow(H_hp, cmap='gray'); axes[1, 2].set_title('High-pass filter')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('demo_fft.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_fft.png")
    plt.close()
    print(f"Low-pass range: [{lp_result.min():.1f}, {lp_result.max():.1f}]")


# ============================================================
# Demo 4: Wiener filter for motion blur restoration
# ============================================================
def demo_wiener_restoration():
    print("=" * 50)
    print("[Demo 4] Wiener filter: Motion blur restoration")
    print("=" * 50)

    img = create_test_image().astype(np.float32)
    blurred, psf = add_motion_blur(img.astype(np.uint8), kernel_size=20, angle=0)
    blurred_noisy = add_gaussian_noise(blurred, sigma=5)

    restored = wiener(blurred_noisy.astype(np.float64), mysize=15, noise=0.01)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    def wiener_freq(blurred, psf, K=0.01):
        h, w = blurred.shape
        psf_pad = np.zeros_like(blurred, dtype=np.float32)
        ph, pw = psf.shape
        psf_pad[:ph, :pw] = psf
        PSF = np.fft.fft2(psf_pad)
        G = np.fft.fft2(blurred.astype(np.float32))
        H_conj = np.conj(PSF)
        W = H_conj / (np.abs(PSF)**2 + K)
        F_hat = W * G
        f_hat = np.abs(np.fft.ifft2(F_hat))
        return np.clip(f_hat, 0, 255).astype(np.uint8)

    restored_freq = wiener_freq(blurred_noisy, psf, K=0.005)

    show_comparison(
        [img.astype(np.uint8), blurred, blurred_noisy, restored, restored_freq],
        ['Original', 'Motion blur', 'Blur+noise', 'scipy Wiener', 'Freq Wiener (K=0.005)'],
        save_path='demo_wiener.png'
    )


# ============================================================
# Demo 5: Canny edge detection
# ============================================================
def demo_canny_edge():
    print("=" * 50)
    print("[Demo 5] Canny edge detection: Optimal edge operator")
    print("=" * 50)

    img = create_test_image()
    img_denoised = cv2.GaussianBlur(img, (3, 3), 0)

    sobel_x = cv2.Sobel(img_denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
    laplacian = np.abs(cv2.Laplacian(img_denoised, cv2.CV_64F)).astype(np.uint8)

    canny_loose = cv2.Canny(img_denoised, 30, 90)
    canny_strict = cv2.Canny(img_denoised, 50, 150)
    canny_tight = cv2.Canny(img_denoised, 80, 240)

    show_comparison(
        [img, sobel, laplacian, canny_loose, canny_strict, canny_tight],
        ['Original', 'Sobel', 'Laplacian', 'Canny(30,90)', 'Canny(50,150) [recommended]', 'Canny(80,240) too tight'],
        figsize=(20, 4),
        save_path='demo_canny.png'
    )

    print("\nCanny 3 steps: Gaussian smooth -> Gradient -> NMS -> Double threshold")
    print("Rule: high/low ratio = 3:1 or 2:1; high = Otsu * 0.7")

    otsu_t, _ = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure otsu_t is a Python scalar (numpy 2.x compatibility)
    otsu_t = int(otsu_t)
    canny_auto = cv2.Canny(img_denoised, int(otsu_t * 0.5), int(otsu_t))
    print(f"Otsu auto thresholds: low={int(otsu_t * 0.5)}, high={int(otsu_t)}")
    show_comparison([canny_strict, canny_auto],
                    ['Manual (50,150)', f'Otsu auto ({int(otsu_t * 0.5)},{int(otsu_t)})'],
                    save_path='demo_canny_auto.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    demo_spatial_filtering()
    demo_sharpening()
    demo_frequency_filtering()
    demo_wiener_restoration()
    demo_canny_edge()
    print("\n[OK] Stage 2 Enhancement & Restoration demo complete!")
