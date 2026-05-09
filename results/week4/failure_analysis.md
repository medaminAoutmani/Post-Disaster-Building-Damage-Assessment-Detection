# Week 4 Failure Case Analysis

Use the prediction panels in `results/week4/predictions/` and the per-sample CSV in
`results/week4/metrics/prediction_records.csv` to compare the pretrained encoder against Week 3.

## Best Predictions

- Samples where building footprints align well with the ground truth:
- Improvements compared with Week 3:

## Difficult Scenes

- Tiny buildings:
- Shadows:
- Smoke or haze:
- Flood reflections:
- Dense urban regions:
- Partial/unclear building boundaries:

## Failure Cases

- False positives:
- False negatives:
- Missed destroyed buildings:
- Mask shifted or fragmented:
- Empty or near-empty predictions:

## Research Directions

- Fine-tune encoder learning rate separately from decoder learning rate.
- Test ResNet50 or EfficientNet encoders.
- Move from binary building segmentation to multiclass damage segmentation.
