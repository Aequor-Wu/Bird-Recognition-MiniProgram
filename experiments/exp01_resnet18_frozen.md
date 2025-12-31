# Experiments Log

## Experiment 1: ResNet18 (Frozen Backbone)

### Environment
- OS: Windows
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Framework: PyTorch
- CUDA: Enabled (torch.cuda.is_available() == True)
- Python: 3.8 (conda environment: birdgpu)

### Dataset
- Name: CUB-200-2011
- Classes: 200
- Input size: 224 × 224
- Split: train / test (official split)
- Data loader: torchvision.datasets.ImageFolder

### Model
- Backbone: ResNet18 (ImageNet pretrained)
- Strategy: Transfer Learning (Frozen Backbone)
- Trainable layers: Final fully connected layer only

### Training Configuration
- Batch size: 8
- Epochs: 10
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss

### Results (Top-1 Accuracy on Validation Set)

| Epoch | Train Loss | Val Accuracy (%) |
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

### Observations
- Training loss decreases steadily, indicating stable convergence.
- Validation accuracy improves rapidly in early epochs and stabilizes around 50%.
- Minor fluctuations in validation accuracy are observed, which is expected due to random initialization and small batch size.
- The frozen-backbone strategy provides reasonable performance with limited training cost.

### Notes
- This experiment serves as the baseline model.
- Next step: compare with a non-frozen (fine-tuned) ResNet18 to analyze the effect of full model training.
