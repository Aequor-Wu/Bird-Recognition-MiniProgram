# Experiment Log
- Batch size: 8
- Epochs: 10

## 1. Experiment Setup

### 1.1 Dataset
- Name: CUB-200-2011
- Number of classes: 200
- Input size: 224 × 224
- Data split: Official train / test split
- Data loader: torchvision.datasets.ImageFolder
- Evaluation metric: Top-1 Accuracy

### 1.2 Model
- Backbone: ResNet18 (ImageNet pretrained)
- Training strategy: Full Fine-tuning
- Trainable layers: All layers (backbone + classification head)
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
- Learning rate: 1e-4
- Batch size: 8
- Epochs: 10

### 2.3 Training Results (Top-1 Accuracy)

| Epoch | Train Loss | Validation Accuracy (%) |
|------:|-----------:|------------------------:|
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

- Final validation accuracy: **67.62%**

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

- Compared with the frozen-backbone baseline (Exp01), full fine-tuning significantly
  improves validation accuracy.
- Fine-tuning allows pretrained features to adapt better to fine-grained bird species
  classification.
- Validation accuracy reaches over 67%, demonstrating the effectiveness of updating
  backbone parameters.
- Minor accuracy fluctuations are observed in later epochs, which may indicate mild
  overfitting.

---

## 5. Notes

- This experiment explores the impact of unfreezing the backbone during training.
- Full fine-tuning achieves substantially higher accuracy than the frozen-backbone
  strategy at the cost of increased computation.
- Future work:
  - Run the same experiment on a server GPU (RTX 5090) for efficiency comparison.
  - Compare training time and convergence behavior across different hardware setups.
