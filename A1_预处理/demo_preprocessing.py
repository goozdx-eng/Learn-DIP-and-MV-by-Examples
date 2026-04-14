"""
第1阶段：图像预处理 OpenCV 实例
涵盖：点运算、直方图均衡化、CLAHE、仿射变换、透视变换

运行环境：pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 工具函数：统一显示多图对比
# ============================================================
def show_images(images, titles, cmap_list=None, figsize=(16, 4)):
    n = len(images)
    if cmap_list is None:
        cmap_list = ['gray'] * n
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title, cmap in zip(axes, images, titles, cmap_list):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# 1. 读取测试图像（无图时自动生成一张模拟车牌灰度图）
# ============================================================
def create_fake_plate(shape=(120, 400)):
    """生成一张模拟低质量车牌：整体偏暗 + 轻微旋转"""
    img = np.zeros(shape, dtype=np.uint8)
    # 绘制背景
    img[:] = 40                              # 整体偏暗
    # 模拟字符区域（白色矩形）
    positions = [20, 70, 120, 170, 230, 280, 330]
    for x in positions:
        cv2.rectangle(img, (x, 20), (x + 40, 100), 200, -1)
    # 添加椒盐噪声模拟脏污
    noise = np.random.randint(0, 50, shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    # 轻微旋转（模拟拍摄角度）
    M = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), 8, 1.0)
    img = cv2.warpAffine(img, M, (shape[1], shape[0]))
    return img


# ============================================================
# 2. 点运算演示
# ============================================================
def demo_point_operations(img_gray):
    print("=" * 50)
    print("【演示1】点运算：线性拉伸 / 对数变换 / 伽马校正")
    print("=" * 50)

    # 线性对比度拉伸
    img_min, img_max = img_gray.min(), img_gray.max()
    linear = ((img_gray.astype(np.float32) - img_min) /
              (img_max - img_min) * 255).astype(np.uint8)

    # 对数变换（提升暗部细节）
    c = 255 / np.log(1 + 255)
    log_img = (c * np.log(1 + img_gray.astype(np.float32))).astype(np.uint8)

    # 伽马校正（γ=0.5 增亮）
    gamma = 0.5
    gamma_img = (255 * (img_gray.astype(np.float32) / 255) ** gamma).astype(np.uint8)

    show_images(
        [img_gray, linear, log_img, gamma_img],
        ['原图（偏暗）', '线性拉伸', '对数变换(提亮暗部)', f'伽马校正(γ={gamma})']
    )


# ============================================================
# 3. 直方图均衡化演示
# ============================================================
def demo_histogram_equalization(img_gray):
    print("=" * 50)
    print("【演示2】直方图均衡化 vs CLAHE（局部自适应）")
    print("=" * 50)

    # 全局直方图均衡化
    eq = cv2.equalizeHist(img_gray)

    # CLAHE：限制对比度自适应均衡化（车牌场景推荐）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_gray)

    # 绘制直方图对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    imgs = [img_gray, eq, clahe_img]
    names = ['原图', '全局均衡化', 'CLAHE']
    for col, (im, name) in enumerate(zip(imgs, names)):
        axes[0, col].imshow(im, cmap='gray')
        axes[0, col].set_title(name)
        axes[0, col].axis('off')
        axes[1, col].hist(im.ravel(), 256, [0, 256], color='steelblue')
        axes[1, col].set_title(f'{name} 直方图')
        axes[1, col].set_xlim([0, 256])
    plt.tight_layout()
    plt.show()

    return clahe_img


# ============================================================
# 4. 仿射变换演示（旋转校正）
# ============================================================
def demo_affine_transform(img_gray):
    print("=" * 50)
    print("【演示3】仿射变换：旋转、平移、缩放")
    print("=" * 50)

    h, w = img_gray.shape

    # 旋转校正（逆向旋转 -8 度）
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, -8, 1.0)
    rotated = cv2.warpAffine(img_gray, M_rot, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

    # 平移（向右20像素、向下10像素）
    M_trans = np.float32([[1, 0, 20], [0, 1, 10]])
    translated = cv2.warpAffine(img_gray, M_trans, (w, h))

    # 缩放到固定尺寸（统一为 100×300）
    resized = cv2.resize(img_gray, (300, 100), interpolation=cv2.INTER_LINEAR)

    show_images(
        [img_gray, rotated, translated, resized],
        ['原图（有旋转）', '旋转校正(-8°)', '平移', '缩放至300×100'],
        figsize=(16, 4)
    )

    return rotated


# ============================================================
# 5. 透视变换演示（梯形 → 矩形）
# ============================================================
def demo_perspective_transform(img_gray):
    print("=" * 50)
    print("【演示4】透视变换：模拟车牌透视畸变校正")
    print("=" * 50)

    h, w = img_gray.shape

    # 模拟透视畸变：把矩形图故意做成梯形
    src_pts = np.float32([[20, 10], [w - 20, 10],
                           [w - 5, h - 5], [5, h - 5]])  # 梯形（模拟斜拍）
    dst_pts = np.float32([[0, 0], [w, 0],
                           [w, h], [0, h]])                # 标准矩形

    # 正向：制造畸变
    M_forward = cv2.getPerspectiveTransform(dst_pts, src_pts)
    distorted = cv2.warpPerspective(img_gray, M_forward, (w, h))

    # 逆向：校正畸变
    M_correct = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(distorted, M_correct, (w, h))

    show_images(
        [img_gray, distorted, corrected],
        ['原图', '模拟透视畸变（梯形）', '透视校正（恢复矩形）'],
        figsize=(15, 5)
    )

    # 打印变换矩阵（便于理解）
    print(f"\n透视变换矩阵 M:\n{np.round(M_correct, 4)}")

    return corrected


# ============================================================
# 6. 完整预处理流水线（模拟车牌场景）
# ============================================================
def full_preprocessing_pipeline(img_gray):
    print("=" * 50)
    print("【演示5】完整预处理流水线（车牌场景）")
    print("=" * 50)
    print("步骤：原图 → CLAHE增强 → 旋转校正 → 透视校正 → 归一化尺寸")

    h, w = img_gray.shape

    # Step1：CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    step1 = clahe.apply(img_gray)

    # Step2：旋转校正
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -8, 1.0)
    step2 = cv2.warpAffine(step1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Step3：透视校正（这里简单用resize模拟）
    step3 = cv2.resize(step2, (400, 120), interpolation=cv2.INTER_LINEAR)

    # Step4：归一化（像素值映射到0~1 float）
    step4_norm = step3.astype(np.float32) / 255.0
    step4_display = (step4_norm * 255).astype(np.uint8)

    show_images(
        [img_gray, step1, step2, step3, step4_display],
        ['原始输入', 'Step1 CLAHE', 'Step2 旋转校正', 'Step3 透视/缩放', 'Step4 归一化'],
        figsize=(20, 4)
    )

    print(f"\n最终输出尺寸：{step3.shape}，数值范围：[{step4_norm.min():.2f}, {step4_norm.max():.2f}]")
    return step3


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # 尝试读取真实图像，失败则使用生成的模拟图
    import os
    test_img_path = 'test_plate.jpg'

    if os.path.exists(test_img_path):
        img_gray = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        print(f"已加载图像：{test_img_path}，尺寸：{img_gray.shape}")
    else:
        img_gray = create_fake_plate()
        print("未找到测试图像，已自动生成模拟车牌图像")
        print("提示：将真实车牌图命名为 test_plate.jpg 放在同目录下可替换")

    # 依次演示各知识点
    demo_point_operations(img_gray)
    enhanced = demo_histogram_equalization(img_gray)
    rotated = demo_affine_transform(img_gray)
    corrected = demo_perspective_transform(img_gray)
    result = full_preprocessing_pipeline(img_gray)

    print("\n✅ 第1阶段预处理演示完成！下一步：进入第2阶段「增强与复原」")
