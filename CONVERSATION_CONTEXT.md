# Conversation Context: Week 11 Object-Level Damage Classification

## Starting Goal

Implement the Week 11 roadmap:

- Freeze the best segmentation baseline.
- Transition from dense damage segmentation to building-level damage classification.
- Extract building instances from xBD.
- Build an object-level crop dataset.
- Train a first Siamese classifier baseline.
- Track building-level metrics and error analysis.
- Introduce morphology/TDA features only after the baseline is established.

## Week 11 Implementation Added

New module:

```text
src/week11/
```

Files added or updated:

```text
src/week11/__init__.py
src/week11/week11_extract_buildings.py
src/week11/week11_dataset.py
src/week11/week11_model.py
src/week11/week11_train_classifier.py
src/week11/week11_error_analysis.py
src/week11/week11_features.py
```

Documentation updated:

```text
README.md
PROJECT_REPORT.md
```

## Building Extraction Pipeline

Script:

```text
src/week11/week11_extract_buildings.py
```

Purpose:

- Reads xBD pre/post images and GT polygon labels.
- Converts one building polygon/component into one object-level sample.
- Uses connected components and minimum-area filtering.
- Saves pre/post/difference crops plus object metadata.

Output structure:

```text
data/week11_buildings/
    train/
        no_damage/
        minor_damage/
        major_damage/
        destroyed/
    val/
    test/
```

Each sample folder contains:

```text
pre.png
post.png
diff.png
mask.png
metadata.json
```

Recommended extraction command:

```bash
python src/week11/week11_extract_buildings.py \
  --output-root data/week11_buildings \
  --crop-size 96 \
  --padding 12 \
  --min-area 32
```

## Dataset Class

Script:

```text
src/week11/week11_dataset.py
```

Dataset:

```text
BuildingDamageDataset
```

Returns:

```text
pre
post
diff
label
class_name
building_id
sample_id
disaster_type
metadata_path
```

Important fix:

- The dataset originally returned nested `metadata`.
- PyTorch DataLoader failed because polygon lists have variable lengths.
- Fix: return `metadata_path`, `sample_id`, and `disaster_type` instead of raw nested metadata.

Later additions:

- `augment=True` applies lightweight crop augmentation.
- `include_features=True` returns handcrafted morphology/TDA features.

## Baseline Classifier

Script:

```text
src/week11/week11_model.py
```

Model:

```text
SiameseBuildingClassifier
```

Architecture:

- Shared ResNet18 encoder.
- Inputs: pre crop, post crop, diff crop.
- Fusion:

```text
concat(features_pre, features_post, features_diff, abs(features_pre - features_post))
```

- MLP head:

```text
Linear -> ReLU -> Dropout -> Linear
```

Loss:

```text
CrossEntropyLoss
```

## Metric Bug Fixed

The first metrics implementation added epsilon to the F1 numerator, causing impossible F1 values when TP was zero.

Example from early run:

```text
minor_damage recall = 0
major_damage recall = 0
but F1 appeared high
```

Fix:

```text
precision = TP / predicted
recall = TP / support
f1 = 2 * TP / (support + predicted)
```

Also added:

```text
predicted_<class>
```

This made it easier to diagnose minority overprediction.

## Initial Baseline Result

Initial baseline:

```bash
python src/week11/week11_train_classifier.py \
  --dataset-root data/week11_buildings \
  --epochs 20 \
  --batch-size 32
```

Dataset counts:

```text
train_samples = 44395
val_samples = 12496

train_class_counts:
no_damage:     40855
minor_damage:    277
major_damage:    172
destroyed:      3091

val_class_counts:
no_damage:     11136
minor_damage:     68
major_damage:     49
destroyed:      1243
```

Conclusion:

- Huge class imbalance remains even after object extraction.
- Initial model learned mostly `no_damage` and `destroyed`.
- `minor_damage` and `major_damage` collapsed to near-zero recall.

Corrected early metrics from confusion matrix:

```text
accuracy:    0.9643
macro F1:    0.4592
weighted F1: 0.9592

no_damage F1:     0.9808
minor_damage F1:  0.0000
major_damage F1:  0.0000
destroyed F1:     0.8558
```

## Imbalance Controls Added

Script:

```text
src/week11/week11_train_classifier.py
```

Added:

```text
--class-weight-mode none|inverse|effective
--weighted-sampler
--augment-train
--max-train-per-class NO_DAMAGE MINOR MAJOR DESTROYED
```

Class caps use `-1` to keep all samples for a class.

## Effective Weighted + Sampler Experiment

Command:

```bash
python src/week11/week11_train_classifier.py \
  --dataset-root data/week11_buildings \
  --results-dir results/week11_balanced_effective_sampler \
  --epochs 20 \
  --batch-size 32 \
  --class-weight-mode effective \
  --weighted-sampler
```

Conclusion:

- Helped `major_damage` somewhat.
- `minor_damage` remained very weak.
- Showed that balancing helps recall but does not solve class separability.

## Capped + Augmented Effective Baseline

Command:

```bash
python src/week11/week11_train_classifier.py \
  --dataset-root data/week11_buildings \
  --results-dir results/week11_capped_augmented \
  --epochs 25 \
  --batch-size 32 \
  --class-weight-mode effective \
  --weighted-sampler \
  --augment-train \
  --max-train-per-class 5000 -1 -1 3091
```

Best checkpoint:

```text
epoch 23
```

Metrics:

```text
accuracy:              0.9431
macro F1:              0.5067
weighted F1:           0.9524

no_damage F1:          0.9703
minor_damage F1:       0.0262
major_damage F1:       0.1561
destroyed F1:          0.8743

minor_damage recall:   0.0441
major_damage recall:   0.4286
destroyed recall:      0.8922
```

Main conclusion:

```text
The object-level classifier substantially improves no-damage and destroyed classification
and begins to recover major-damage buildings under capped, augmented training. However,
minor damage remains poorly separable, suggesting that the main bottleneck is not
architecture but class scarcity and visual ambiguity.
```

This is the current best Week 11 baseline.

## Inverse Weighted Experiment

Command:

```bash
python src/week11/week11_train_classifier.py \
  --dataset-root data/week11_buildings \
  --results-dir results/week11_capped_augmented_inverse \
  --epochs 25 \
  --batch-size 32 \
  --class-weight-mode inverse \
  --weighted-sampler \
  --augment-train \
  --max-train-per-class 3000 -1 -1 3091
```

Best checkpoint:

```text
epoch 25
```

Metrics:

```text
accuracy:              0.5739
macro F1:              0.3855
weighted F1:           0.7145

minor_damage recall:   0.4412
minor_damage precision:0.0115
minor_damage F1:       0.0224
predicted_minor:       2606
support_minor:         68

major_damage recall:   0.6327
major_damage precision:0.0120
major_damage F1:       0.0236
predicted_major:       2574
support_major:         49
```

Conclusion:

```text
Inverse class weighting over-corrects the imbalance problem. It improves minority recall,
but only by flooding the validation set with minor and major predictions, causing very low
precision and a large drop in overall accuracy and weighted F1.
```

Therefore:

```text
results/week11_capped_augmented
```

remains the stronger baseline.

## Error Analysis Interpretation

After error analysis, the project conclusion should be:

```text
The remaining failure is not mainly a model-capacity problem. It is a
class-definition and data-distribution problem.
```

Expected failure taxonomy:

```text
minor -> no_damage: subtle roof changes, visually ambiguous
minor -> major: small debris/cracks look severe
major -> destroyed: heavy roof loss, high-severity overlap
destroyed -> major: partial structure remains
```

Report conclusion:

```text
Error analysis shows that the object-level classifier correctly learns visually distinct
classes such as no-damage and destroyed buildings. Capped and augmented training improves
major-damage recall, confirming that object-level crops contain useful damage cues.
However, minor damage remains poorly separated because it is rare, subtle, and visually
close to both no-damage and major-damage cases. Therefore, the next bottleneck is not
architecture search, but minority-class data quality, label ambiguity, and feature
enrichment.
```

## Phase 6 Implemented: Morphology/TDA Feature Fusion

Script:

```text
src/week11/week11_features.py
```

Feature categories:

- Morphology
- Change statistics
- Lightweight topology
- Optional Gudhi persistent homology

Feature examples:

```text
mask_area_ratio
metadata_area_log
perimeter_log
compactness
bbox_aspect_ratio
bbox_area_ratio
extent
solidity
contour_count
hole_count
euler_number
contour_fragmentation
largest_component_ratio
distance_mean
distance_std
distance_max
edge_density_pre
edge_density_post
edge_density_delta
diff_mean
diff_std
diff_p90
diff_p95
diff_high_ratio
diff_component_count
diff_largest_component_ratio
diff_euler_number
diff_contour_fragmentation
```

Optional Gudhi features:

```text
mask_ph_dim0_count
mask_ph_dim0_entropy
mask_ph_dim0_mean_lifetime
mask_ph_dim1_count
mask_ph_dim1_entropy
diff_ph_dim0_count
diff_ph_dim0_entropy
diff_ph_dim1_count
diff_ph_dim1_entropy
```

If Gudhi is not installed, persistent-homology fields are zeros, and the pipeline still runs.

Install optional full TDA dependency:

```bash
pip install gudhi
```

## Feature-Fusion Model

Model:

```text
SiameseBuildingFeatureClassifier
```

Defined in:

```text
src/week11/week11_model.py
```

Fusion:

```text
final_feature = concat(
    cnn_pre,
    cnn_post,
    cnn_diff,
    abs(cnn_pre - cnn_post),
    morphology_tda_embedding
)
```

Enable with:

```text
--use-handcrafted-features
```

Recommended Phase 6 run:

```bash
python src/week11/week11_train_classifier.py \
  --dataset-root data/week11_buildings \
  --results-dir results/week11_feature_fusion \
  --epochs 25 \
  --batch-size 32 \
  --class-weight-mode effective \
  --weighted-sampler \
  --augment-train \
  --max-train-per-class 5000 -1 -1 3091 \
  --use-handcrafted-features
```

Feature-fusion error analysis:

```bash
python src/week11/week11_error_analysis.py \
  --dataset-root data/week11_buildings \
  --checkpoint results/week11_feature_fusion/checkpoints/week11_siamese_resnet18_best.pt \
  --output-dir results/week11_feature_fusion/error_analysis \
  --use-handcrafted-features
```

## Current Scientific Position

The best current baseline is:

```text
results/week11_capped_augmented
```

The inverse-weighted run is a negative result.

The next research question is:

```text
Do morphology, change, and topology/TDA features improve minority-class precision
without destroying minority recall?
```

Compare:

```text
results/week11_capped_augmented
vs
results/week11_feature_fusion
```

Key metrics to compare:

```text
macro F1
minor_damage precision
minor_damage recall
minor_damage F1
major_damage precision
major_damage recall
major_damage F1
destroyed F1
predicted_minor_damage
predicted_major_damage
```

## Validation Performed

Local checks run:

```text
python -m compileall src/week11
git diff --check
```

Status:

```text
compileall passed
git diff --check passed
```

Only normal Windows LF/CRLF warnings appeared.

