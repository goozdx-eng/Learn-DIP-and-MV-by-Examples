"""
Stage 5: Classification - OpenCV Examples
Covers: SVM (linear/RBF), ANN/MLP, AdaBoost, Cross-validation, Confusion matrix

Install: pip install opencv-python numpy matplotlib scikit-learn scikit-image
Run: python demo_classification.py
"""

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Generate feature dataset
# ============================================================
def generate_feature_dataset():
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        raise ImportError("Install scikit-image: pip install scikit-image")

    font = cv2.FONT_HERSHEY_SIMPLEX
    char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    samples, labels = [], []
    np.random.seed(42)

    for char in char_set:
        for _ in range(30):
            img = np.ones((80, 40), dtype=np.uint8) * 200
            scale = 1.5 + np.random.uniform(-0.15, 0.15)
            thick = np.random.randint(2, 4)
            dx, dy = np.random.randint(-3, 3), np.random.randint(-3, 3)
            brightness = int(np.random.uniform(0.5, 1.0) * 200)
            img[:] = brightness
            cv2.putText(img, char, (5+dx, 65+dy), font, scale, 30, thick)
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
            feat = hist.astype(np.float32)
            feat /= (feat.sum() + 1e-6)
            samples.append(feat)
            labels.append(char)

    X = np.array(samples)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    print(f"Char set: {''.join(le.classes_)}")
    return X, y, le, labels


# ============================================================
# Demo 1: SVM (sklearn + OpenCV)
# ============================================================
def demo_svm(X_train, X_test, y_train, y_test, le):
    print("=" * 50)
    print("[Demo 1] SVM: Linear kernel vs RBF kernel + Grid search")
    print("=" * 50)

    results = {}

    for kernel, params in [('linear', {'C': 1}), ('rbf', {'C': 10, 'gamma': 0.01})]:
        clf = SVC(kernel=kernel, **params, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[f'SVM-{kernel}'] = acc
        print(f"SVM ({kernel} kernel, {params}): accuracy = {acc*100:.2f}%")

    svm_cv = cv2.ml.SVM_create()
    svm_cv.setType(cv2.ml.SVM_C_SVC)
    svm_cv.setKernel(cv2.ml.SVM_RBF)
    svm_cv.setC(10)
    svm_cv.setGamma(0.01)
    svm_cv.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm_cv.train(X_train.astype(np.float32),
                  cv2.ml.ROW_SAMPLE,
                  y_train.astype(np.int32))
    _, y_pred_cv = svm_cv.predict(X_test.astype(np.float32))
    acc_cv = accuracy_score(y_test, y_pred_cv.flatten().astype(int))
    results['OpenCV-SVM-RBF'] = acc_cv
    print(f"OpenCV SVM (RBF kernel): accuracy = {acc_cv*100:.2f}%")

    print("\nRunning grid search for optimal SVM parameters (C, gamma)...")
    param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_acc = grid_search.score(X_test, y_test)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best SVM accuracy: {best_acc*100:.2f}%")
    results['SVM-GridSearch'] = best_acc
    return grid_search.best_estimator_, results


# ============================================================
# Demo 2: ANN/MLP
# ============================================================
def demo_ann(X_train, X_test, y_train, y_test, le):
    print("=" * 50)
    print("[Demo 2] ANN/MLP Neural Network Classification")
    print("=" * 50)

    results = {}

    configs = [
        {'hidden_layer_sizes': (64,), 'name': 'MLP-1layer-64'},
        {'hidden_layer_sizes': (128, 64), 'name': 'MLP-2layer-128-64'},
        {'hidden_layer_sizes': (256, 128, 64), 'name': 'MLP-3layer-256-128-64'},
    ]
    for cfg in configs:
        name = cfg.pop('name')
        mlp = MLPClassifier(max_iter=500, random_state=42, **cfg)
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        results[name] = acc
        print(f"{name}: accuracy = {acc*100:.2f}%")

    n_classes = len(le.classes_)
    n_features = X_train.shape[1]

    ann = cv2.ml.ANN_MLP_create()
    layer_sizes = np.array([n_features, 128, 64, n_classes], dtype=np.int32)
    ann.setLayerSizes(layer_sizes)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 500, 1e-4))

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
    print(f"OpenCV ANN_MLP (128-64): accuracy = {acc_ann*100:.2f}%")

    best_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    best_mlp.fit(X_train, y_train)
    plt.figure(figsize=(10, 4))
    plt.plot(best_mlp.loss_curve_, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.title('MLP(128-64) Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig('demo_ann_loss.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_ann_loss.png")
    plt.close()
    return best_mlp, results


# ============================================================
# Demo 3: AdaBoost
# ============================================================
def demo_adaboost(X_train, X_test, y_train, y_test):
    print("=" * 50)
    print("[Demo 3] AdaBoost: Weak classifier ensemble boosting")
    print("=" * 50)

    results = {}
    n_estimators_list = [10, 25, 50, 100]

    for n in n_estimators_list:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n, learning_rate=0.5, random_state=42
        )
        ada.fit(X_train, y_train)
        acc = ada.score(X_test, y_test)
        results[f'AdaBoost-{n}'] = acc
        print(f"AdaBoost (n={n:3d}): accuracy = {acc*100:.2f}%")

    accs = [results[f'AdaBoost-{n}'] for n in n_estimators_list]
    plt.figure(figsize=(8, 4))
    plt.plot(n_estimators_list, [a*100 for a in accs], 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Number of weak classifiers')
    plt.ylabel('Test accuracy (%)')
    plt.title('AdaBoost: Accuracy vs Number of Iterations')
    plt.grid(True, alpha=0.3)
    plt.savefig('demo_adaboost.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_adaboost.png")
    plt.close()

    best_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=50, learning_rate=0.5, random_state=42
    )
    best_ada.fit(X_train, y_train)
    return best_ada, results


# ============================================================
# Demo 4: Comprehensive comparison + confusion matrix
# ============================================================
def demo_comparison_and_confusion(X_train, X_test, y_train, y_test, le,
                                   all_results, best_svm, best_ann, best_ada):
    print("=" * 50)
    print("[Demo 4] Classifier comparison + Confusion matrix (best model)")
    print("=" * 50)

    flat_results = {}
    for d in all_results:
        flat_results.update(d)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    names = list(flat_results.keys())
    accs = [flat_results[n] * 100 for n in names]
    colors = ['#e74c3c' if 'SVM' in n else '#3498db' if 'ANN' in n or 'MLP' in n
              else '#2ecc71' for n in names]
    bars = axes[0].barh(names, accs, color=colors, alpha=0.85)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Classifier Accuracy Comparison')
    axes[0].set_xlim([0, 105])
    for bar, acc in zip(bars, accs):
        axes[0].text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                     f'{acc:.1f}%', va='center', fontsize=9)

    best_name = max(flat_results, key=flat_results.get)
    print(f"\nBest classifier: {best_name} ({flat_results[best_name]*100:.2f}%)")

    y_pred_best = best_svm.predict(X_test)

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
        axes[1].set_xlabel('Predicted label')
        axes[1].set_ylabel('True label')
        axes[1].set_title('Digit Confusion Matrix (best SVM model)')
        for i in range(len(char_names)):
            for j in range(len(char_names)):
                axes[1].text(j, i, str(cm[i, j]),
                              ha='center', va='center',
                              color='white' if cm[i, j] > cm.max()*0.5 else 'black',
                              fontsize=9)

    plt.tight_layout()
    plt.savefig('demo_comparison.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_comparison.png")
    plt.close()

    print("\nDetailed classification report (first 10 classes):")
    y_pred_all = best_svm.predict(X_test)
    report = classification_report(y_test, y_pred_all,
                                    target_names=le.classes_,
                                    labels=list(range(min(10, len(le.classes_)))))
    print(report)


# ============================================================
# Demo 5: Full pipeline
# ============================================================
def demo_full_pipeline(best_svm, le):
    print("=" * 50)
    print("[Demo 5] Complete license plate recognition pipeline summary")
    print("=" * 50)

    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        print("Requires scikit-image")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    def predict_char(char_img, svm_model, le):
        char_resized = cv2.resize(char_img, (40, 80))
        lbp = local_binary_pattern(char_resized, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        feat = hist.astype(np.float32)
        feat /= (feat.sum() + 1e-6)
        pred_idx = svm_model.predict([feat])[0]
        return le.inverse_transform([pred_idx])[0]

    test_chars = list('BA1234C')
    print(f"True char sequence: {''.join(test_chars)}")

    char_imgs, predictions = [], []
    for char in test_chars:
        img = np.ones((80, 40), dtype=np.uint8) * 195
        scale = 1.5 + np.random.uniform(-0.1, 0.1)
        cv2.putText(img, char, (5, 65), font, scale, 30, 3)
        noise = np.random.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        char_imgs.append(img)
        pred = predict_char(img, best_svm, le)
        predictions.append(pred)

    print(f"Predicted:         {''.join(predictions)}")
    correct = sum(t == p for t, p in zip(test_chars, predictions))
    print(f"Accuracy: {correct}/{len(test_chars)} = {correct/len(test_chars)*100:.1f}%")

    fig, axes = plt.subplots(2, len(test_chars), figsize=(16, 5))
    for i, (img, true_c, pred_c) in enumerate(zip(char_imgs, test_chars, predictions)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Input', fontsize=9)
        axes[0, i].axis('off')

        color = '#2ecc71' if true_c == pred_c else '#e74c3c'
        axes[1, i].text(0.5, 0.5, pred_c, ha='center', va='center',
                         fontsize=36, fontweight='bold', color=color,
                         transform=axes[1, i].transAxes)
        mark = '[OK]' if true_c == pred_c else '[X]'
        axes[1, i].set_title(f'True:{true_c} Pred:{pred_c} {mark}', fontsize=9,
                              color=color)
        axes[1, i].axis('off')
        axes[1, i].set_facecolor('#f8f9fa')

    plt.suptitle(f'Pipeline result: {"".join(predictions)} ({correct}/{len(test_chars)} correct)',
                  fontsize=13)
    plt.tight_layout()
    plt.savefig('demo_pipeline_result.png', dpi=150, bbox_inches='tight')
    print("[Saved] -> demo_pipeline_result.png")
    plt.close()

    print("\nPipeline summary:")
    print("  Original -> [Geom transform + CLAHE] -> [Median/Gaussian filter] -> [Otsu binarize] ->")
    print("  [Morphological open] -> [Connected components cut] -> [Resize 40x80] ->")
    print("  [LBP histogram 59 dims] -> [SVM classify] -> Character result")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating feature dataset...")
    X, y, le, labels = generate_feature_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    best_svm, svm_results = demo_svm(X_train, X_test, y_train, y_test, le)
    best_ann, ann_results = demo_ann(X_train, X_test, y_train, y_test, le)
    best_ada, ada_results = demo_adaboost(X_train, X_test, y_train, y_test)

    demo_comparison_and_confusion(
        X_train, X_test, y_train, y_test, le,
        [svm_results, ann_results, ada_results],
        best_svm, best_ann, best_ada)

    demo_full_pipeline(best_svm, le)
    print("\n[OK] Stage 5 Classification demo complete!")
    print("=" * 60)
    print(">> Complete license plate recognition pipeline done!")
    print("   Preprocessing -> Enhancement -> Segmentation -> Feature -> Classification")
    print("=" * 60)
