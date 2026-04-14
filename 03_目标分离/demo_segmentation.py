"""
第3阶段：目标分离 OpenCV 实例
涵盖：二值化（Otsu/自适应）、形态学操作（腐蚀/膨胀/开闭/顶帽）、连通域分析、车牌字符切割

运行环境：pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 生成模拟车牌（含边框、字符、铆钉干扰）
# ============================================================
def create_license_plate(h=120, w=400):
    """生成带干扰的模拟车牌二值图"""
    img = np.ones((h, w), dtype=np.uint8) * 200  # 浅灰背景

    # 外边框
    cv2.rectangle(img, (3, 3), (w-4, h-4), 50, 3)

    # 字符（白色背景，深色字）
    font = cv2.FONT_HERSHEY_SIMPLEX
    chars = ['B', 'A', '1', '2', '3', '4', 'C']
    for i, ch in enumerate(chars):
        x = 20 + i * 54
        cv2.putText(img, ch, (x, 90), font, 1.6, 30, 3)

    # 模拟铆钉（小圆点噪声）
    for pos in [(15, 60), (385, 60), (15, 30), (385, 30)]:
        cv2.circle(img, pos, 6, 30, -1)

    # 添加轻微高斯噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def show_images(imgs, titles, cmap='gray', figsize=(18, 4)):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1: axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# 演示1：二值化方法对比
# ============================================================
def demo_binarization(img):
    print("=" * 50)
    print("【演示1】二值化：全局 / Otsu / 自适应")
    print("=" * 50)

    # 全局固定阈值
    _, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Otsu 自动阈值
    otsu_thresh, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 自适应高斯阈值（块大小=21）
    binary_adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)

    show_images(
        [img, binary_fixed, binary_otsu, binary_adaptive],
        ['原图', '固定阈值127', f'Otsu(阈值={otsu_thresh:.0f})', '自适应(块=21)']
    )
    print(f"Otsu自动找到的最优阈值：{otsu_thresh:.0f}")
    return binary_otsu


# ============================================================
# 演示2：形态学基本操作
# ============================================================
def demo_morphology_basics(binary):
    print("=" * 50)
    print("【演示2】形态学操作：腐蚀、膨胀、开运算、闭运算")
    print("=" * 50)

    se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    se5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded = cv2.erode(binary, se3)         # 腐蚀：白色缩小
    dilated = cv2.dilate(binary, se3)       # 膨胀：白色扩大
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se5)    # 开：去小噪点
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se5)   # 闭：填小孔洞

    show_images(
        [binary, eroded, dilated, opened, closed],
        ['二值原图', '腐蚀(3×3)', '膨胀(3×3)', '开运算(5×5)去噪', '闭运算(5×5)填孔']
    )

    # 高级操作
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, se5)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, se5)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, se3)

    show_images(
        [tophat, blackhat, gradient],
        ['顶帽（亮小目标）', '黑帽（暗小目标）', '形态学梯度（轮廓）']
    )


# ============================================================
# 演示3：去除铆钉噪声（开运算实战）
# ============================================================
def demo_remove_rivets(binary):
    print("=" * 50)
    print("【演示3】开运算去除铆钉/小噪点（字符保留，小点消除）")
    print("=" * 50)

    # 铆钉大约6个像素半径，字符宽度约20像素
    # SE高度设为10，可以消除小于10的噪点但保留字符
    se_rivet = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_rivet)

    # 连通域分析显示效果
    num_before, _, stats_before, _ = cv2.connectedComponentsWithStats(binary)
    num_after, _, stats_after, _ = cv2.connectedComponentsWithStats(cleaned)

    show_images([binary, cleaned], ['开运算前（含铆钉）', '开运算后（铆钉消除）'])
    print(f"开运算前连通域数量：{num_before - 1}（背景不计）")
    print(f"开运算后连通域数量：{num_after - 1}（背景不计）")

    return cleaned


# ============================================================
# 演示4：连通域分析 + 字符切割
# ============================================================
def demo_connected_components(img, binary):
    print("=" * 50)
    print("【演示4】连通域分析：逐字符切割与过滤")
    print("=" * 50)

    h, w = img.shape

    # 开运算先去噪
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

    print(f"总连通域数（含背景）：{num_labels}")
    print(f"{'标签':>4} {'X':>5} {'Y':>5} {'W':>5} {'H':>5} {'面积':>7} {'保留?':>6}")
    print("-" * 45)

    # 筛选字符区域：面积在合理范围内，高宽比合适
    char_regions = []
    min_area = 200
    max_area = h * w * 0.15   # 不超过总面积的15%
    min_h = h * 0.3           # 高度至少是车牌高度的30%

    for i in range(1, num_labels):   # 跳过标签0（背景）
        x, y, bw, bh, area = stats[i][:5]
        keep = (min_area < area < max_area) and (bh > min_h)
        mark = "✓" if keep else "✗"
        print(f"{i:>4} {x:>5} {y:>5} {bw:>5} {bh:>5} {area:>7} {mark:>6}")
        if keep:
            char_regions.append((x, y, bw, bh, area))

    # 按x坐标排序（从左到右）
    char_regions.sort(key=lambda r: r[0])

    # 可视化：在原图上画边界框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 128)]
    for idx, (x, y, bw, bh, _) in enumerate(char_regions):
        color = colors[idx % len(colors)]
        cv2.rectangle(img_color, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(img_color, str(idx+1), (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 显示切割结果
    fig, axes = plt.subplots(2, max(1, len(char_regions)), figsize=(16, 5))
    if len(char_regions) == 1:
        axes = axes.reshape(2, 1)

    # 第一行：带框原图
    ax_main = plt.subplot2grid((2, len(char_regions)), (0, 0),
                                colspan=len(char_regions))
    ax_main.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f'连通域分析结果（共检出 {len(char_regions)} 个字符）')
    ax_main.axis('off')

    # 第二行：每个字符单独显示
    plt.figure(figsize=(16, 3))
    for idx, (x, y, bw, bh, _) in enumerate(char_regions):
        char_img = img[y:y+bh, x:x+bw]
        char_resized = cv2.resize(char_img, (40, 80))
        plt.subplot(1, len(char_regions), idx+1)
        plt.imshow(char_resized, cmap='gray')
        plt.title(f'字符{idx+1}')
        plt.axis('off')
    plt.suptitle('切割出的字符（已归一化到40×80）')
    plt.tight_layout()
    plt.show()

    return char_regions


# ============================================================
# 演示5：轮廓检测（另一种分割方式）
# ============================================================
def demo_contour_detection(img, binary):
    print("=" * 50)
    print("【演示5】轮廓检测：findContours + 矩特征")
    print("=" * 50)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)

    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

    print(f"检测到轮廓数量：{len(contours)}")
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # 最小外接矩形（可处理旋转）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 图像矩（用于计算重心）
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img_color, (cx, cy), 3, (0, 0, 255), -1)
        print(f"  轮廓{i}: 面积={area:.0f}, 中心=({cx},{cy}), 最小外接矩形={rect[1]}")

    plt.figure(figsize=(14, 4))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('轮廓检测（绿线）+ 重心（红点）')
    plt.axis('off')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # 生成模拟车牌
    plate = create_license_plate()
    print(f"车牌尺寸：{plate.shape}，灰度范围：[{plate.min()}, {plate.max()}]")

    # 依次演示
    binary = demo_binarization(plate)
    demo_morphology_basics(binary)
    cleaned_binary = demo_remove_rivets(binary)
    char_regions = demo_connected_components(plate, cleaned_binary)
    demo_contour_detection(plate, cleaned_binary)

    print(f"\n最终切割出 {len(char_regions)} 个字符区域")
    print("\n✅ 第3阶段目标分离演示完成！下一步：进入第4阶段「特征提取」")
