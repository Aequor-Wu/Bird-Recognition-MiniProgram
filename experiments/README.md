# Experiments Overview

This folder records all training and evaluation experiments
for the bird species recognition project.

Each experiment ID corresponds to a specific model and training strategy.
Different runs under the same experiment may vary by hardware environment
or training configuration.

## Experiment Summary

| Exp ID | Model     | Strategy          | Environment        | Val Acc (%) | Notes |
|-------|-----------|-------------------|--------------------|-------------|------|
| Exp01 | ResNet18  | Frozen Backbone   | Local GPU (RTX3050)| ~50.1       | Baseline run |
| Exp01 | ResNet18  | Frozen Backbone   | Server GPU (RTX5090)| TBD        | Planned |
| Exp02 | ResNet18  | Full Fine-tuning  | Local GPU (RTX3050)| TBD         | Planned |
| Exp02 | ResNet18  | Full Fine-tuning  | Server GPU (RTX5090)| TBD        | Planned |

## Experiment Logs

- `exp01_resnet18_frozen.md`  
  ResNet18 transfer learning experiment with frozen backbone.
  Includes results from both local and server GPU runs.

- `exp02_resnet18_finetune.md`  
  ResNet18 full fine-tuning experiment.

