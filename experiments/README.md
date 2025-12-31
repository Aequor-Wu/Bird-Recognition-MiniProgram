# Experiments Overview

This folder records all training and evaluation experiments
for the bird species recognition project.

Each experiment ID corresponds to a specific model and training strategy.
Different runs under the same experiment may vary by hardware environment
or training configuration.

## Experiment Summary

| Exp ID | Model    | Strategy         | Environment | Batch | LR     | Epochs | Val Acc (%) | Notes |
|-------|----------|------------------|-------------|-------|--------|--------|-------------|------|
| Exp01 | ResNet18 | Frozen Backbone  | RTX3050     | 8     | 0.001  | 10     | 50.14       | Baseline (local GPU) |
| Exp01 | ResNet18 | Frozen Backbone  | RTX5090     | TBD   | TBD    | TBD    | TBD         | Server run |
| Exp02 | ResNet18 | Full Fine-tuning | RTX3050     | TBD   | TBD    | TBD    | TBD         | Local comparison |
| Exp02 | ResNet18 | Full Fine-tuning | RTX5090     | TBD   | TBD    | TBD    | TBD         | Server comparison |


## Experiment Logs

- `exp01_resnet18_frozen.md`  
  ResNet18 transfer learning experiment with frozen backbone.
  Includes results from both local and server GPU runs.

- `exp02_resnet18_finetune.md`  
  ResNet18 full fine-tuning experiment.

Note: Training hyperparameters may vary across environments due to hardware constraints.

