# 数据集说明（Dataset Description）

本项目使用 **CUB-200-2011 数据集**
（Caltech-UCSD Birds-200-2011），
该数据集是细粒度鸟类分类（Fine-grained Bird Classification）领域中
被广泛采用的标准数据集之一。

---

## 1. 数据集基本信息

- 数据集名称：CUB-200-2011  
- 数据集类型：细粒度图像分类数据集（Fine-grained Image Classification）  
- 鸟类类别数量（Number of classes）：200  
- 图像总数量（Number of images）：11,788  
- 标注类型：图像级类别标签（Image-level class labels）

每张图片对应一个明确的鸟类物种标签，
适用于监督学习（Supervised Learning）任务。

---

## 2. 数据集特点

- 各类别之间外观差异较小，属于典型的**细粒度分类问题**  
- 同一类别内部存在姿态、光照、背景等变化  
- 对模型的特征提取能力和泛化能力要求较高  

因此，该数据集非常适合用于评估：
- 迁移学习（Transfer Learning）
- 预训练模型（Pretrained Models）
- 微调策略（Fine-tuning Strategy）

在鸟类识别和细粒度视觉任务中具有较高研究价值。

---

## 3. 数据划分方式

本项目采用 CUB-200-2011 官方提供的数据划分方式：

- 训练集（Train set）
- 测试集（Test set）

数据加载过程中使用
`torchvision.datasets.ImageFolder`，
根据文件夹结构自动分配类别标签。

---

## 4. 数据集来源

数据集下载与整理来源如下：

- GitHub 仓库地址：  
  https://github.com/cyizhuo/CUB_200_2011_dataset

---

## 5. 本项目中的使用说明

在本项目中，CUB-200-2011 数据集用于：

- ResNet18 迁移学习（冻结骨干网络）
- ResNet18 全模型微调（Fine-tuning）
- 不同硬件环境（本地 GPU 与服务器 GPU）下的性能对比实验

该数据集为后续模型设计、训练策略选择及实验分析
提供了统一且可靠的数据基础。

