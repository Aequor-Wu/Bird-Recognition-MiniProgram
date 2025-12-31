# 实验记录（Exp01）：ResNet18 冻结骨干网络（Frozen Backbone）

## 一、实验目的

本实验旨在评估 **ResNet18** 在 **冻结骨干网络（Frozen Backbone）** 的迁移学习（Transfer Learning）策略下，  
在 **CUB-200-2011 鸟类细粒度分类数据集** 上的分类性能。

该实验作为后续 **全模型微调（Full Fine-tuning）实验（Exp02）** 的**基线实验（Baseline）**。

---

## 二、数据集说明（Dataset）

- 数据集名称：CUB-200-2011  
- 类别数量（Number of Classes）：200  
- 输入尺寸（Input Size）：224 × 224  
- 数据划分方式：官方提供的 train / test 划分  
- 数据加载方式（Data Loader）：  
  `torchvision.datasets.ImageFolder`  
  - 根据文件夹结构自动分配类别标签  
- 评估指标（Evaluation Metric）：  
  **Top-1 Accuracy**  
  - 表示模型预测概率最高的类别是否与真实标签一致  

---

## 三、模型与训练策略（Model & Strategy）

- 主干网络（Backbone）：ResNet18（ImageNet 预训练）  
- 训练策略（Training Strategy）：迁移学习（Transfer Learning）  
- 参数冻结方式：
  - 冻结 ResNet18 的所有卷积层参数  
  - 仅训练最后一层全连接层（Fully Connected Layer）  
- 损失函数（Loss Function）：CrossEntropyLoss  

---

## 四、本地 GPU 实验（Local GPU：RTX 3050）

### 4.1 实验环境（Environment）

- 操作系统（OS）：Windows  
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU  
- 深度学习框架（Framework）：PyTorch  
- CUDA：启用（`torch.cuda.is_available() == True`）  
- Python 版本：3.8  
- Conda 环境：`birdgpu`  

---

### 4.2 训练配置（Training Configuration）

- 优化器（Optimizer）：Adam  
- 学习率（Learning Rate）：1e-3  
- Batch Size：8  
- 训练轮数（Epochs）：10  

---

### 4.3 实验结果（Top-1 Accuracy）

| Epoch | Train Loss | 验证集准确率 (%) |
|------:|-----------:|-----------------:|
| 1 | 4.2940 | 33.62 |
| 2 | 2.5622 | 38.87 |
| 3 | 1.9914 | 46.19 |
| 4 | 1.7045 | 48.17 |
| 5 | 1.4917 | 48.33 |
| 6 | 1.3676 | 48.29 |
| 7 | 1.2313 | 49.36 |
| 8 | 1.1523 | 48.58 |
| 9 | 1.0999 | 50.69 |
| 10 | 0.9981 | 50.14 |

- 最终验证集准确率（Final Validation Accuracy）：**50.14%**

---

## 五、服务器 GPU 实验（Server GPU：RTX 5090）

### 5.1 实验环境（Environment）

- 操作系统（OS）：Ubuntu  
- GPU：NVIDIA RTX 5090  
- 深度学习框架（Framework）：PyTorch  
- CUDA：启用  
- Python 版本：3.8  

---

### 5.2 训练配置（Training Configuration）

- 优化器（Optimizer）：Adam  
- 学习率（Learning Rate）：TBD  
- Batch Size：TBD  
- Epochs：TBD  

---

### 5.3 实验结果（Top-1 Accuracy）

> 待补充（Planned）

---

## 六、实验结果分析（Analysis）

- 训练损失（Training Loss）随 Epoch 稳定下降，说明模型训练过程较为稳定，未出现明显震荡。  
- 验证集准确率在前几个 Epoch 内快速提升，随后逐渐趋于平稳，表明模型较快完成特征适配。  
- 验证准确率在约 50% 附近波动，属于小 Batch Size 与细粒度分类任务中常见现象。  
- 冻结骨干网络的策略在显著降低训练成本的同时，仍能取得较为合理的分类性能。  

---

## 七、备注与后续计划（Notes & Future Work）

- 本实验作为后续实验的**基线模型（Baseline）**。  
- 实验结果表明：  
  ImageNet 预训练的视觉特征对鸟类细粒度分类任务具有较好的迁移能力。  
- 后续工作计划：
  - 在服务器 GPU（RTX 5090）上复现实验，用于训练效率对比。  
  - 开展 **Exp02：ResNet18 全模型微调（Full Fine-tuning）实验**，分析解冻骨干网络对性能的提升效果。  


