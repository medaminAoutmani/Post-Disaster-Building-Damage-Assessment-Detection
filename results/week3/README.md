# Week 3 Results

This folder is the reproducibility package for the improved Week 3 segmentation baseline.

After running `src/week3_train.py`, it should contain:

```text
results/week3/
+-- checkpoints/
+-- dataset_statistics/
+-- figures/
+-- logs/
+-- predictions/
+-- failure_analysis.md
```

Expected artifacts:

- `checkpoints/week3_unet_binary_best.pt`: best model weights
- `logs/training_log.csv`: epoch-by-epoch training and validation metrics
- `logs/best_metrics.json`: best validation metrics
- `logs/run_config.json`: training configuration
- `logs/prediction_records.csv`: saved qualitative samples with Dice and category
- `dataset_statistics/week3_train_dataset_statistics.csv`: cleaned dataset summary
- `figures/loss_curve.png`: train/validation loss curve
- `figures/dice_curve.png`: train/validation Dice curve
- `predictions/epoch_XXX/`: input, ground-truth, prediction panels
- `failure_analysis.md`: notes for qualitative failure analysis

Qualitative panels are grouped into:

- `good_predictions/`
- `difficult_scenes/`
- `failure_cases/`
