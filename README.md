# Learn DIP & MV by Examples

```
╔═══════════════════════════════════════════════════════════════════════╗
║  DIGITAL IMAGE PROCESSING & MACHINE VISION                              ║
║  From Concept Intuition to Engineering Practice                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Foundation (S1~S9)  →  Application (A1~A5)  →  Advanced (OPEN)        ║
║  Concept × Why         Engineering × How        Competition × Paper     ║
╚═══════════════════════════════════════════════════════════════════════╝
```

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg?style=flat-square&logo=opencv"/>
  <img src="https://img.shields.io/badge/Matlab-Simulink-orange.svg?style=flat-square&logo=mathworks"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/goozdx-eng/Learn-DIP-and-MV-by-Examples?style=flat-square"/>
</p>

---

## 痛点

课本写的是"概念 → 定义 → 公式"，你背下来去考试，卷子做完全忘。这不是你的问题——教材是给"已经理解了"的人当参考手册用的，不是给"第一次学"的人当教程用的。

这份教程的逻辑反过来：**问题 → 困惑 → 原理 → 验证 → 坑点**。每个知识点从工程现场的真实困惑出发，推导设计原因，让概念变成可以推理的直觉。代码demo让你跑完"看见"这个概念在图上的实际表现。

---

## 流水线

```
[信号采集]
    │
[数字化] ─── S3采样 × S3量化 × S8插值
    │              ↓
[类型定义] ─── S2二值 / S2灰度 / S2RGB / S2索引
    │
[颜色模型] ─── S4RGB / S4HSI / S4YUV
    │
[文件格式] ─── S5BMP / S5PNG / S5JPG ⚠ 中间数据禁止JPG
    │
[S6点运算] ─── 灰度反转 / 阈值分割 / S7直方图均衡化
    │
[S6邻域运算] ── 线性滤波（均值/高斯）── 非线性（中值/双边）
    │
[频域分析] ─── A2傅里叶 / 低通 / 高通 / 带通
    │
[边缘检测] ─── S9梯度 / Sobel / Canny
    │
[目标分割] ─── A3Otsu / 形态学 / 分水岭
    │
[特征提取] ─── A4HOG / LBP / SIFT / ORB
    │
[分类识别] ─── A5SVM / ANN / AdaBoost
    │
[几何变换] ─── A1旋转 / 缩放 / 插值 / 配准
```

---

## 章节结构

```
├── 通识术语.md                ← 学习路径导航，按顺序读
├── README.md                  ← 本文件
│
├── [Foundation] 基础层（S前缀，9章）
│   ├── S1_图像基础概念/       像素 / 灰度级 / 位深 / 分辨率
│   ├── S2_图像类型/           二值 / 灰度 / RGB / 索引
│   ├── S3_数字化过程/         采样 / 量化 / 摩尔纹
│   ├── S4_颜色模型/           RGB / HSI / HSV / YUV
│   ├── S5_文件格式/           BMP / PNG / JPG / PGM
│   ├── S6_运算类型/           点 / 邻域 / 线性 / 非线性
│   ├── S7_直方图/             均衡化 / 规定化 / CDF
│   ├── S8_几何变换/           插值 / 图像配准
│   └── S9_边缘检测基础/        梯度 / 阈值分割 / Otsu
│
├── [Application] 应用层（A前缀，5章）
│   ├── A1_预处理/             滤波选型 / 去噪评估 / 几何校正
│   ├── A2_增强与复原/         空域增强 / 频域增强 / 维纳滤波
│   ├── A3_目标分离/          阈值分割 / 分水岭 / 区域生长
│   ├── A4_特征提取/           纹理 / 形状 / 角点 / 描述符
│   └── A5_分类识别/           KNN / SVM / ANN / 混淆矩阵
│
└── [Advanced] 高级层
    └── （竞赛真题 / 论文复现 / 项目实战，开放提交）
```

每章包含：
- **教程.md** — 原理讲解（困惑→类比→原理→坑点）
- **demo_*.py** — 配套OpenCV代码，跑完看见效果

---

## 学习方式

每章开头有"前置警告"，是学习这章的门槛条件。每章结尾有**检验题**，做不对说明基础概念还没过关，不要急着往下走。

> 如果读完一章没有"啊，原来是这样"的感受，说明还没读透——重读检验题对应的章节段落，直到能独立回答所有检验题。

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/goozdx-eng/Learn-DIP-and-MV-by-Examples.git
cd Learn-DIP-and-MV-by-Examples

# 安装依赖
pip install opencv-python numpy matplotlib

# 运行示例
python S1_图像基础概念/demo_pixel.py
python S6_运算类型/demo_operation_types.py
python S9_边缘检测基础/demo_edge.py
```

---

## 教学设计原则

| 步骤 | 内容 | 目的 |
|:---|:---|:---|
| 1. 困惑 | 开篇提出真实工程问题 | 建立问题意识 |
| 2. 类比 | 用生活实例建立直觉 | 把抽象概念挂到已知经验上 |
| 3. 原理 | 数学/物理逻辑推导 | 理解"为什么这样设计" |
| 4. 验证 | 检验题测试理解深度 | 确认能独立推导 |
| 5. 坑点 | 常见错误和易混概念 | 避免在同一个坑里跌倒两次 |

---

## 典型问题 → 章节映射

| 你遇到的问题 | 对应章节 |
|:---|:---|
| 拍屏幕有摩尔纹 | S3 数字化过程 - 采样定理 |
| 边缘检测噪点多 | S9 边缘检测 + A1 预处理 |
| 颜色识别不稳定 | S4 颜色模型 - HSI vs RGB |
| 处理后图变模糊 | S6 运算类型 - 线性滤波的代价 |
| 旋转后有锯齿 | S8 几何变换 - 插值方法 |
| 直方图均衡化帮倒忙 | S7 直方图 - 全局 vs 自适应 |
| 中间数据用JPG变差 | S5 文件格式 - JPG有损压缩 |
| 索引图颜色不对 | S2 图像类型 - 调色板查询 |
| 去噪后边缘变钝 | A1 预处理 - 双边滤波选型 |
| 分割结果粘连严重 | A3 目标分离 - 分水岭优化 |

---

<p align="center">
Built with 数字图像处理 · 齐鲁工业大学 · 2026
</p>
