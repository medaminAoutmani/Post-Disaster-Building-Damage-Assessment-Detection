# Week 3 Failure Case Analysis

Use the prediction panels in `results/week3/predictions/` and the per-sample CSV in `results/week3/metrics/prediction_records.csv` to document visual patterns.

## Best Predictions

- Samples where building footprints align well with the ground truth:
- Common scene properties:

## Difficult Scenes

- Tiny buildings:
- Shadows:
- Smoke or haze:
- Flood reflections:
- Dense urban regions:
- Partial or unclear building boundaries:

## Failure Cases

- False positives:
- False negatives:
- Missed destroyed buildings:
- Mask shifted or fragmented:
- Empty or near-empty predictions:

## Research Directions

- More robust augmentation for shadows, smoke, and flood reflections.
- Higher-resolution crops for tiny buildings.
- Multiclass damage segmentation instead of binary building segmentation.
- Attention U-Net or pretrained encoder U-Net.
- Temporal change encoding between pre-disaster and post-disaster imagery.
