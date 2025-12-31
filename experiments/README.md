# 实验总览（Experiments Overview）

本文件夹用于记录鸟类识别小程序项目中的
**模型训练与性能评估实验（Training & Evaluation Experiments）**。

每一个实验编号（Exp ID）对应一种明确的：
- 模型结构（Model）
- 训练策略（Training Strategy）

在相同实验编号下，可能会因为 **硬件环境（GPU）**
或 **训练参数配置** 的不同，产生多次实验记录，
用于对比分析不同条件下的模型表现。

---

## 实验汇总表（Experiment Summary）

| 实验编号 | 模型 | 训练策略 | 运行环境 | Batch Size | Learning Rate | Epochs | 验证集准确率 (%) | 备注 |
|---------|------|----------|----------|------------|---------------|--------|------------------|------|
| Exp01 | ResNet18 | 冻结骨干网络（Frozen Backbone） | 本地 GPU（RTX 3050） | 8 | 1e-3 | 10 | 50.14 | 基线实验（本地） |
| Exp01 | ResNet18 | 冻结骨干网络（Frozen Backbone） | 服务器 GPU（RTX 5090） | TBD | TBD | TBD | TBD | 计划运行 |
| Exp02 | ResNet18 | 全模型微调（Full Fine-tuning） | 本地 GPU（RTX 3050） | 8 | 1e-4 | 10 | 67.62 | 与基线对比 |
| Exp02 | ResNet18 | 全模型微调（Full Fine-tuning） | 服务器 GPU（RTX 5090） | TBD | TBD | TBD | TBD | 计划运行 |

---

## 实验日志说明（Experiment Logs）

- `exp01_resnet18_frozen.md`  
  ResNet18 冻结骨干网络（Frozen Backbone）迁移学习实验记录。  
  作为整个项目的**基线模型（Baseline）**，用于后续实验对比。

- `exp02_resnet18_finetune.md`  
  ResNet18 全模型微调（Full Fine-tuning）实验记录。  
  在 Exp01 基础上解冻主干网络参数，用于分析微调策略对性能的提升效果。

---

## 说明与备注（Notes）

- 不同实验环境下（如本地 GPU 与服务器 GPU），  
  训练超参数可能会因显存容量与计算能力不同而进行调整。
- Exp01 与 Exp02 使用相同的数据集与模型结构，  
  主要区别在于 **训练策略（是否冻结骨干网络）**。
- 本实验记录将作为后续：
  - 模型优化分析  
  - 开题报告与毕业论文实验章节  
  的重要依据。


