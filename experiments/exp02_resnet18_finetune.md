# 实验记录（Exp02）：ResNet18 全模型微调（Full Fine-tuning）

## 一、实验目的

本实验在 **Exp01（冻结骨干网络）** 的基础上，  
进一步评估 **ResNet18 全模型微调（Full Fine-tuning）** 策略在  
**CUB-200-2011 鸟类细粒度分类数据集** 上的性能表现。

通过解冻主干网络参数，使预训练特征能够更充分地适配鸟类细粒度分类任务，  
并与冻结骨干网络策略进行对比分析。

---

## 二、数据集说明（Dataset）

- 数据集名称：CUB-200-2011  
- 类别数量（Number of Classes）：200  
- 输入尺寸（Input Size）：224 × 224  
- 数据划分方式：官方提供的 train / test 划分  
- 数据加载方式（Data Loader）：  
  `torchvision.datasets.ImageFolder`  
- 评估指标（Evaluation Metric）：  
  **Top-1 Accuracy**  
  - 表示模型预测概率最高的类别是否与真实标签一致  

---

## 三、模型与训练策略（Model & Strategy）

- 主干网络（Backbone）：ResNet18（ImageNet 预训练）  
- 训练策略（Training Strategy）：全模型微调（Full Fine-tuning）  
- 参数更新方式：
  - 主干网络（Backbone）与分类头（Classification Head）全部参与训练  
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
- 学习率（Learning Rate）：1e-4  
- Batch Size：8  
- 训练轮数（Epochs）：10  

---

### 4.3 实验结果（Top-1 Accuracy）

| Epoch | Train Loss | 验证集准确率 (%) |
|------:|-----------:|-----------------:|
| 1 | 4.1387 | 38.99 |
| 2 | 2.4487 | 54.44 |
| 3 | 1.5687 | 62.25 |
| 4 | 1.0063 | 64.93 |
| 5 | 0.6399 | 67.28 |
| 6 | 0.4390 | 68.52 |
| 7 | 0.2814 | 66.98 |
| 8 | 0.1977 | 66.05 |
| 9 | 0.1444 | 68.14 |
| 10 | 0.1200 | 67.62 |

- 最终验证集准确率（Final Validation Accuracy）：**67.62%**

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

- 与 **Exp01 冻结骨干网络实验** 相比，全模型微调策略显著提升了验证集准确率。  
- 全模型训练允许预训练特征进一步适配鸟类细粒度差异，提高了模型判别能力。  
- 验证集准确率最高超过 68%，最终稳定在约 67% 左右，明显优于基线模型。  
- 在后期 Epoch 中出现轻微波动，可能与数据规模有限及轻微过拟合有关。  

---

## 七、备注与后续计划（Notes & Future Work）

- 本实验验证了解冻骨干网络对细粒度分类任务的有效性。  
- 相较冻结骨干网络策略，全模型微调在计算开销增加的情况下，显著提升了分类性能。  
- 后续工作计划：
  - 在服务器 GPU（RTX 5090）上复现实验，用于训练效率与收敛速度对比。  
  - 对比不同硬件环境下的训练时间、显存占用与性能表现。  

