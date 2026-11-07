# M3: Faster R-CNN ResNet50-FPN V2 Pipeline

This directory contains the implementation and training pipeline for **Method 3 (M3)** as defined in the COMP9517 Fixed Training and Evaluation Protocol.

## Configuration & Hyperparameters
- **Model:** Faster R-CNN ResNet50-FPN V2 with official COCO weights (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`).
- **Classes:** 2 classes (Background = 0, Bird = 1).
- **Training Length:** 100 epochs (early stopping disabled).
- **Batch Size:** Actual batch size = 2, Gradient accumulation = 2 (Effective batch size = 4).
- **Optimizer:** SGD (Momentum = 0.9, Nesterov = off).
- **Learning Rate:** Peak LR = 0.0025, Weight Decay = 0.0001, 1 warmup epoch.
- **Precision:** AMP FP16.
- **Backbone:** `trainable_backbone_layers = 3` with frozen BatchNorm stats and affine parameters.
