# Experiment Log

## Exp01: ResNet18 with Frozen Backbone

This experiment evaluates the performance of a ResNet18 model using a frozen backbone
transfer learning strategy on the CUB-200-2011 bird species dataset.
This experiment serves as the baseline for subsequent fine-tuning experiments.

---

## 1. Experiment Setup

### 1.1 Dataset
- Name: CUB-200-2011
- Number of classes: 200
- Input size: 224 × 224
- Data split: Official train / test split
- Data loader: torchvision.datasets.ImageFolder, which automatically assigns class labels based on directory structure.
- Evaluation metric: Top-1 Accuracy, which measures whether the model’s most confident prediction matches the ground-truth label.


### 1.2 Model
- Backbone: ResNet18 (ImageNet pretrained)
- Training strategy: Transfer Learning (Frozen Backbone)
- Trainable layers: Final fully connected layer only
- Loss function: CrossEntropyLoss

---

## 2. Local GPU Experiment (RTX 3050)

### 2.1 Environment
- OS: Windows
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Framework: PyTorch
- CUDA: Enabled (`torch.cuda.is_available() == True`)
- Python: 3.8
- Conda environment: `birdgpu`

### 2.2 Training Configuration
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 8
- Epochs: 10


### 2.3 Training Results (Top-1 Accuracy)

| Epoch | Train Loss | Validation Accuracy (%) |
|------:|-----------:|------------------------:|
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

- Final validation accuracy: **50.14%**

---
## 3. Server GPU Experiment (RTX 5090)

### 3.1 Environment
- OS: Ubuntu
- GPU: NVIDIA RTX 5090
- Framework: PyTorch
- CUDA: Enabled
- Python: 3.8

### 3.2 Training Configuration
- Optimizer: Adam
- Learning rate: TBD
- Batch size: TBD
- Epochs: TBD

### 3.3 Training Results (Top-1 Accuracy)


---
## 4. Analysis

- Training loss decreases steadily across epochs, indicating stable convergence.
- Validation accuracy improves rapidly during early epochs and gradually saturates.
- Minor fluctuations in validation accuracy are observed, which is expected given the
  relatively small batch size and dataset complexity.
- Freezing the backbone significantly reduces training cost while still achieving
  reasonable classification performance.

---

## 5. Notes

- This experiment serves as the **baseline** for subsequent experiments.
- The frozen-backbone strategy demonstrates that pretrained visual features from ImageNet
  transfer effectively to fine-grained bird species classification.
- Future work:
  - Run the same experiment on a server GPU (RTX 5090) for efficiency comparison.
  - Compare with **Exp02: ResNet18 with full fine-tuning** to analyze the effect of
    unfreezing the backbone.

