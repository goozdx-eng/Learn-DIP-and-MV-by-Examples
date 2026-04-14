"""
第5阶段：分类识别 OpenCV 实例
涵盖：SVM（线性/RBF核）、ANN/MLP、AdaBoost、交叉验证、混淆矩阵

运行环境：pip install opencv-python numpy matplotlib scikit-learn scikit-image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 数据准备（复用第4阶段的特征，这里直接生成）
# ============================================================
def generate_feature_dataset():
    """生成包含LBP特征的字符数据集"""
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        raise ImportError("请安装scikit-image：pip install scikit-image")

    font = cv2.FONT_HERSHEY_SIMPLEX
    # 10个数字 + 26个字母，模拟真实车牌字符集
    char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    samples, labels = [], []
    np.random.seed(42)

    for char in char_set:
        for _ in range(30):  # 每类30个样本（适度数据）
            img = np.ones((80, 40), dtype=np.uint8) * 200
            scale = 1.5 + np.random.uniform(-0.15, 0.15)
            thick = np.random.randint(2, 4)
            dx, dy = np.random.randint(-3, 3), np.random.randint(-3, 3)
            # 随机光照
            brightness = int(np.random.uniform(0.5, 1.0) * 200)
            img[:] = brightness
            cv2.putText(img, char, (5+dx, 65+dy), font, scale, 30, thick)
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 提取LBP直方图特征
            lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
            feat = hist.astype(np.float32)
            feat /= (feat.sum() + 1e-6)

            samples.append(feat)
            labels.append(char)

    X = np.array(samples)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"数据集：{X.shape[0]} 样本，{X.shape[1]} 特征维，{len(le.classes_)} 类")
    print(f"字符集：{''.join(le.classes_)}")

    return X, y, le, labels


# ============================================================
# 演示1：SVM 分类（sklearn + OpenCV两种实现）
# ============================================================
def demo_svm(X_train, X_test, y_train, y_test, le):
    print("=" * 50)
    print("【演示1】SVM 分类：线性核 vs RBF核 + 参数调优")
    print("=" * 50)

    results = {}

    # sklearn SVM（功能完整，调试方便）
    for kernel, params in [('linear', {'C': 1}), ('rbf', {'C': 10, 'gamma': 0.01})]:
        clf = SVC(kernel=kernel, **params, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[f'SVM-{kernel}'] = acc
        print(f"SVM ({kernel}核，{params})：准确率 = {acc*100:.2f}%")

    # ---- OpenCV SVM ----
    svm_cv = cv2.ml.SVM_create()
    svm_cv.setType(cv2.ml.SVM_C_SVC)
    svm_cv.setKernel(cv2.ml.SVM_RBF)
    svm_cv.setC(10)
    svm_cv.setGamma(0.01)
    svm_cv.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    # OpenCV SVM 需要 float32 输入和 int32 标签
    svm_cv.train(X_train.astype(np.float32),
                  cv2.ml.ROW_SAMPLE,
                  y_train.astype(np.int32))

    _, y_pred_cv = svm_cv.predict(X_test.astype(np.float32))
    acc_cv = accuracy_score(y_test, y_pred_cv.flatten().astype(int))
    results['OpenCV-SVM-RBF'] = acc_cv
    print(f"OpenCV SVM (RBF核)：准确率 = {acc_cv*100:.2f}%")

    # 网格搜索最优参数
    print("\n正在网格搜索最优SVM参数（C, γ）...")
    param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_acc = grid_search.score(X_test, y_test)
    print(f"最优参数：{grid_search.best_params_}")
    print(f"最优SVM准确率：{best_acc*100:.2f}%")
    results['SVM-GridSearch'] = best_acc

    return grid_search.best_estimator_, results


# ============================================================
# 演示2：ANN/MLP 分类
# ============================================================
def demo_ann(X_train, X_test, y_train, y_test, le):
    print("=" * 50)
    print("【演示2】ANN/MLP 神经网络分类")
    print("=" * 50)

    results = {}

    # sklearn MLP（用于快速验证）
    configs = [
        {'hidden_layer_sizes': (64,), 'name': 'MLP-1层64'},
        {'hidden_layer_sizes': (128, 64), 'name': 'MLP-2层128-64'},
        {'hidden_layer_sizes': (256, 128, 64), 'name': 'MLP-3层256-128-64'},
    ]
    for cfg in configs:
        name = cfg.pop('name')
        mlp = MLPClassifier(max_iter=500, random_state=42, **cfg)
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        results[name] = acc
        print(f"{name}：准确率 = {acc*100:.2f}%")

    # ---- OpenCV ANN_MLP ----
    n_classes = len(le.classes_)
    n_features = X_train.shape[1]

    ann = cv2.ml.ANN_MLP_create()
    # 网络结构：输入-隐层-输出
    layer_sizes = np.array([n_features, 128, 64, n_classes], dtype=np.int32)
    ann.setLayerSizes(layer_sizes)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                          500, 1e-4))

    # 转换标签为one-hot编码（OpenCV ANN要求）
    def to_onehot(y, n_classes):
        onehot = np.zeros((len(y), n_classes), dtype=np.float32)
        onehot[np.arange(len(y)), y] = 1.0
        return onehot

    y_train_oh = to_onehot(y_train, n_classes)
    ann.train(X_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train_oh)

    _, y_pred_ann = ann.predict(X_test.astype(np.float32))
    y_pred_labels = np.argmax(y_pred_ann, axis=1)
    acc_ann = accuracy_score(y_test, y_pred_labels)
    results['OpenCV-ANN_MLP'] = acc_ann
    print(f"OpenCV ANN_MLP (128-64)：准确率 = {acc_ann*100:.2f}%")

    # 训练损失曲线（sklearn MLP）
    best_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                              random_state=42, verbose=False)
    best_mlp.fit(X_train, y_train)
    plt.figure(figsize=(10, 4))
    plt.plot(best_mlp.loss_curve_, 'b-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('训练损失')
    plt.title('MLP(128-64) 训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_mlp, results


# ============================================================
# 演示3：AdaBoost 分类
# ============================================================
def demo_adaboost(X_train, X_test, y_train, y_test):
    print("=" * 50)
    print("【演示3】AdaBoost：弱分类器集成提升")
    print("=" * 50)

    results = {}
    n_estimators_list = [10, 25, 50, 100]

    for n in n_estimators_list:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n, learning_rate=0.5, random_state=42,
            algorithm='SAMME'
        )
        ada.fit(X_train, y_train)
        acc = ada.score(X_test, y_test)
        results[f'AdaBoost-{n}'] = acc
        print(f"AdaBoost (n={n:3d})：准确率 = {acc*100:.2f}%")

    # 绘制准确率 vs 迭代轮数
    accs = [results[f'AdaBoost-{n}'] for n in n_estimators_list]
    plt.figure(figsize=(8, 4))
    plt.plot(n_estimators_list, [a*100 for a in accs], 'g-o', linewidth=2, markersize=8)
    plt.xlabel('弱分类器数量')
    plt.ylabel('测试集准确率 (%)')
    plt.title('AdaBoost：迭代轮数对准确率的影响')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    best_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=50, learning_rate=0.5, random_state=42,
        algorithm='SAMME'
    )
    best_ada.fit(X_train, y_train)
    return best_ada, results


# ============================================================
# 演示4：综合对比 + 混淆矩阵
# ============================================================
def demo_comparison_and_confusion(X_train, X_test, y_train, y_test, le,
                                   all_results, best_svm, best_ann, best_ada):
    print("=" * 50)
    print("【演示4】分类器综合对比 + 混淆矩阵（最优模型）")
    print("=" * 50)

    # 汇总准确率对比
    flat_results = {}
    for d in all_results:
        flat_results.update(d)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    names = list(flat_results.keys())
    accs = [flat_results[n] * 100 for n in names]
    colors = ['#e74c3c' if 'SVM' in n else '#3498db' if 'ANN' in n or 'MLP' in n
              else '#2ecc71' for n in names]
    bars = axes[0].barh(names, accs, color=colors, alpha=0.85)
    axes[0].set_xlabel('准确率 (%)')
    axes[0].set_title('分类器准确率综合对比')
    axes[0].set_xlim([0, 105])
    for bar, acc in zip(bars, accs):
        axes[0].text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                     f'{acc:.1f}%', va='center', fontsize=9)

    # 最优模型的混淆矩阵（取准确率最高的）
    best_name = max(flat_results, key=flat_results.get)
    print(f"\n最优分类器：{best_name}（{flat_results[best_name]*100:.2f}%）")

    # 用SVM-GridSearch做混淆矩阵示例（假设它最优）
    y_pred_best = best_svm.predict(X_test)

    # 只展示数字类别的混淆矩阵（10×10，更清晰）
    digit_classes = [le.transform([c])[0] for c in '0123456789' if c in le.classes_]
    mask = np.isin(y_test, digit_classes)
    if mask.sum() > 0:
        y_t_sub = y_test[mask]
        y_p_sub = y_pred_best[mask]
        cm = confusion_matrix(y_t_sub, y_p_sub, labels=digit_classes)
        char_names = [le.inverse_transform([c])[0] for c in digit_classes]

        im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=axes[1])
        axes[1].set_xticks(range(len(char_names)))
        axes[1].set_yticks(range(len(char_names)))
        axes[1].set_xticklabels(char_names)
        axes[1].set_yticklabels(char_names)
        axes[1].set_xlabel('预测类别')
        axes[1].set_ylabel('真实类别')
        axes[1].set_title('数字类混淆矩阵（SVM最优模型）')
        for i in range(len(char_names)):
            for j in range(len(char_names)):
                axes[1].text(j, i, str(cm[i, j]),
                              ha='center', va='center',
                              color='white' if cm[i, j] > cm.max()*0.5 else 'black',
                              fontsize=9)

    plt.tight_layout()
    plt.show()

    # 详细分类报告（只展示部分类）
    print("\nSVM最优模型详细分类报告（前10类）：")
    y_pred_all = best_svm.predict(X_test)
    report = classification_report(y_test, y_pred_all,
                                    target_names=le.classes_,
                                    labels=list(range(min(10, len(le.classes_)))))
    print(report)


# ============================================================
# 演示5：完整车牌识别流水线
# ============================================================
def demo_full_pipeline(best_svm, le):
    print("=" * 50)
    print("【演示5】完整车牌识别流水线总结")
    print("=" * 50)

    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        print("需要scikit-image")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    def predict_char(char_img, svm_model, le):
        """单个字符图像 → 预测字符"""
        # 归一化尺寸
        char_resized = cv2.resize(char_img, (40, 80))
        # LBP特征
        lbp = local_binary_pattern(char_resized, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        feat = hist.astype(np.float32)
        feat /= (feat.sum() + 1e-6)
        # SVM预测
        pred_idx = svm_model.predict([feat])[0]
        # SVC的predict_proba需要probability=True，这里用decision_function代替
        return le.inverse_transform([pred_idx])[0]

    # 生成测试车牌序列
    test_chars = list('BA1234C')
    print(f"测试字符序列：{''.join(test_chars)}")

    char_imgs = []
    predictions = []
    for char in test_chars:
        # 生成含噪声的字符图
        img = np.ones((80, 40), dtype=np.uint8) * 195
        scale = 1.5 + np.random.uniform(-0.1, 0.1)
        cv2.putText(img, char, (5, 65), font, scale, 30, 3)
        noise = np.random.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        char_imgs.append(img)

        pred = predict_char(img, best_svm, le)
        predictions.append(pred)

    print(f"预测结果：    {''.join(predictions)}")
    correct = sum(t == p for t, p in zip(test_chars, predictions))
    print(f"准确率：{correct}/{len(test_chars)} = {correct/len(test_chars)*100:.1f}%")

    # 可视化
    fig, axes = plt.subplots(2, len(test_chars), figsize=(16, 5))
    for i, (img, true_c, pred_c) in enumerate(zip(char_imgs, test_chars, predictions)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'输入图像', fontsize=9)
        axes[0, i].axis('off')

        color = '#2ecc71' if true_c == pred_c else '#e74c3c'
        axes[1, i].text(0.5, 0.5, pred_c, ha='center', va='center',
                         fontsize=36, fontweight='bold', color=color,
                         transform=axes[1, i].transAxes)
        mark = '✓' if true_c == pred_c else '✗'
        axes[1, i].set_title(f'真实:{true_c} 预测:{pred_c} {mark}', fontsize=9,
                              color=color)
        axes[1, i].axis('off')
        axes[1, i].set_facecolor('#f8f9fa')

    plt.suptitle(f'完整流水线识别结果：{"".join(predictions)}  ({correct}/{len(test_chars)}正确)',
                  fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n流水线总结：")
    print("  原图 → [几何变换+CLAHE] → [中值/高斯滤波] → [Otsu二值化] →")
    print("  [形态学开运算] → [连通域分析切割] → [归一化40×80] →")
    print("  [LBP直方图特征59维] → [SVM分类] → 字符结果")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("生成特征数据集...")
    X, y, le, labels = generate_feature_dataset()

    # 数据集划分（7:3）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练集：{X_train.shape[0]}，测试集：{X_test.shape[0]}")

    # 各分类器演示
    best_svm, svm_results = demo_svm(X_train, X_test, y_train, y_test, le)
    best_ann, ann_results = demo_ann(X_train, X_test, y_train, y_test, le)
    best_ada, ada_results = demo_adaboost(X_train, X_test, y_train, y_test)

    # 综合对比
    demo_comparison_and_confusion(
        X_train, X_test, y_train, y_test, le,
        [svm_results, ann_results, ada_results],
        best_svm, best_ann, best_ada)

    # 完整流水线演示
    demo_full_pipeline(best_svm, le)

    print("\n✅ 第5阶段分类识别演示完成！")
    print("=" * 60)
    print("🎯 整个车牌识别学习路线完成！")
    print("   预处理 → 增强复原 → 目标分离 → 特征提取 → 分类识别")
    print("=" * 60)
