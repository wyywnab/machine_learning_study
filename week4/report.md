# Weekly Result (One-Pager)

## Environment Information


| Item | Version |
|------|---------|
| PyTorch | 2.8.0+cu129 |
| TorchVision | 0.23.0+cu129 |
| CUDA | 12.9 |
| Driver | 581.08 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |


## Summary Table

| exp_id           | exp_name                  |   seed |   epochs |   learning_rate | optimizer   | lr_scheduler      | data_enhancement   | cbam_enabled   |   top-1_acc |   duration |
|:-----------------|:--------------------------|-------:|---------:|----------------:|:------------|:------------------|:-------------------|:---------------|------------:|-----------:|
| exp251014_161144 | baseline_cbam             |    608 |       70 |          0.0001 | AdamW       | CosineAnnealingLR | nan                | True           |      0.7892 |    6830.93 |
| exp251014_181251 | baseline                  |    608 |       75 |          0.0001 | AdamW       | CosineAnnealingLR | nan                | False          |      0.7873 |    6910.66 |
| exp251014_200933 | baseline_enhancement      |    608 |       64 |          0.0001 | AdamW       | CosineAnnealingLR | affine             | False          |      0.9034 |    6087.78 |
| exp251015_084809 | baseline_enhancement_cbam |    608 |       67 |          0.0001 | AdamW       | CosineAnnealingLR | affine             | True           |      0.8883 |    6538.06 |


## Curves

### Validation Accuracy

![*Figure: Validation accuracy across all experiments0*](figs\val_acc.png)

### Loss Curves for baseline_enhancement

![*Figure: Training and validation loss for experiment 'baseline_enhancement'*](figs\loss_baseline_enhancement.png)