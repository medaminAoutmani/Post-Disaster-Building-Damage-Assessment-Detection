# Week 3 Binary Segmentation Baseline

## Objective

Detect building regions from xBD post-disaster imagery using paired pre-disaster and post-disaster satellite images.

This baseline is now preserved as the first stable Week 3 benchmark. Future model changes should be compared against this result instead of overwriting it.

## Model

- Architecture: U-Net
- Input: 6 channels, created by stacking pre-disaster RGB and post-disaster RGB images
- Output: 1-channel binary building mask
- Loss: BCE + Dice
- Optimizer: AdamW
- Target mode: binary segmentation

## Dataset

- Dataset: xBD disaster dataset
- Invalid/noisy samples are filtered before training
- `un-classified` labels are ignored
- Empty masks are skipped

## Results

The final metrics are saved in:

```text
metrics/final_metrics.json
```

Current benchmark target values:

| Metric | Value |
|---|---:|
| Dice | 0.6946 |
| IoU | 0.5512 |
| Precision | 0.7190 |
| Recall | 0.7125 |

## Observations

- Model learns stable building localization.
- Precision and recall are reasonably balanced.
- Training is stable with no catastrophic overfitting.
- Some false positives are expected in dense urban areas.
- Difficult cases may include tiny buildings, shadows, smoke, flood reflections, and ambiguous roof boundaries.

## Folder Structure

```text
results/week3/
+-- checkpoints/
+-- metrics/
+-- predictions/
+-- visualizations/
+-- notebooks/
+-- config/
+-- README.md
```

## Key Artifacts

- `checkpoints/week3_unet_binary_best.pt`: best model weights
- `metrics/final_metrics.json`: best validation metrics
- `metrics/training_log.csv`: epoch-by-epoch training log
- `metrics/dataset_statistics.csv`: cleaned dataset statistics
- `metrics/prediction_records.csv`: qualitative sample categories and Dice values
- `config/training_config.yaml`: reproducible training configuration
- `visualizations/loss_curve.png`: training and validation loss curve
- `visualizations/dice_curve.png`: training and validation Dice curve
- `visualizations/iou_curve.png`: training and validation IoU curve
- `predictions/best_examples/`: strongest qualitative predictions
- `predictions/failure_cases/`: low-Dice failure cases
- `failure_analysis.md`: qualitative analysis notes

## Next Steps

- Multi-class damage segmentation
- Attention U-Net
- Deep supervision
- Temporal change encoding
- Pretrained encoder backbone
