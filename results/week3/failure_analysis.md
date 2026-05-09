# Week 3 Failure Case Analysis

<<<<<<< HEAD
Use the prediction panels in `results/week3/predictions/` and the per-sample CSV in `results/week3/metrics/prediction_records.csv` to document visual patterns.
=======
Use the prediction panels in `results/week3/predictions/` and the per-sample CSV in
`results/week3/metrics/prediction_records.csv` to document visual patterns.
>>>>>>> bbb10aa (executing Training week2 week3 week4)

## Best Predictions

- Samples where building footprints align well with the ground truth:
- Common scene properties:

## Difficult Scenes

- Tiny buildings:
- Shadows:
- Smoke or haze:
- Flood reflections:
- Dense urban regions:
<<<<<<< HEAD
- Partial or unclear building boundaries:
=======
- Partial/unclear building boundaries:
>>>>>>> bbb10aa (executing Training week2 week3 week4)

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
<<<<<<< HEAD
- Attention U-Net or pretrained encoder U-Net.
- Temporal change encoding between pre-disaster and post-disaster imagery.
=======
- Pretrained encoder U-Net for better feature extraction.
>>>>>>> bbb10aa (executing Training week2 week3 week4)
