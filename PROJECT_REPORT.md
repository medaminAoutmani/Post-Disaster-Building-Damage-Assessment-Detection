# Damage Detection Project Report

## Project Description

This project builds a satellite-image damage detection pipeline using the xBD disaster dataset. The main goal is to identify building regions from paired pre-disaster and post-disaster satellite images, then prepare the system for reliable segmentation-based disaster damage analysis.

The dataset contains:

- Pre-disaster RGB satellite images
- Post-disaster RGB satellite images
- JSON label files containing building polygons
- Damage labels such as `no-damage`, `minor-damage`, `major-damage`, `destroyed`, and `un-classified`

The project begins with binary building segmentation, where the model learns whether each pixel belongs to a labeled building or background. It then extends the same pipeline into multiclass damage segmentation, temporal Siamese modeling, class-imbalance handling, and multi-task learning. In the early models, the pre-disaster and post-disaster images are stacked together as a 6-channel input. In the later Siamese models, the two images are processed as separate RGB streams and fused at the feature level. Week 11 transitions from dense pixel-level segmentation to object-level building damage classification. Week 12 extends that transition into embedding-centric building-level representation learning. Week 13 adds a topology-guided verifier for the specific semantic ambiguity between `no_damage` and `minor_damage`. Week 14 adds a CrisisMMD v2 Twitter module for text-based crisis understanding, humanitarian reasoning, damage-severity proxy modeling, and emotion-aware report generation.

The current pipeline is divided into fourteen development stages:

1. Week 1: Data exploration, preprocessing, visualization, and mask generation
2. Week 2: Dataset pipeline, train/validation/test splits, and the first baseline U-Net
3. Week 3: A separate improved baseline with cleaner data, stronger training, prediction visualization, overfit testing, and better metrics
4. Week 4: U-Net upgrade with a pretrained ResNet34 encoder
5. Week 5: Multiclass disaster damage segmentation
6. Week 6: Research experiment framework, ablation studies, upgraded architectures, advanced losses, samplers, schedulers, and visual analysis
7. Week 7: Temporal Siamese damage segmentation with explicit pre/post feature fusion and attention modules
8. Week 8: Rare-class imbalance audit and targeted data expansion for minority damage classes
9. Week 9: Multi-task Siamese segmentation with auxiliary pre-disaster and post-disaster building supervision
10. Week 10: Building-aware damage losses that focus optimization on meaningful damage pixels
11. Week 11: Object-level building extraction and Siamese building damage classification
12. Week 12: Advanced object-level representation learning for semantic damage separability
13. Week 13: Topology-guided semantic calibration for no-damage/minor-damage ambiguity
14. Week 14: CrisisMMD v2 social-media crisis understanding and emotion-aware humanitarian context

## Week 1: Preprocessing and Data Understanding

The first week focused on understanding the xBD data format and converting the raw labels into useful computer vision targets.

The main script for this stage is:

```text
src/week1_preprocessing.py
```

### Work Completed

The Week 1 preprocessing code loads paired satellite images and their matching label files. Each sample has a pre-disaster image, a post-disaster image, and a post-disaster JSON annotation file.

The JSON label files contain building polygons in WKT format. These polygons are parsed using Shapely, converted into OpenCV-compatible coordinate arrays, and then rasterized into segmentation masks.

The project defines the following damage classes:

| Class | Mask ID |
|---|---:|
| background | 0 |
| no-damage | 1 |
| minor-damage | 2 |
| major-damage | 3 |
| destroyed | 4 |

The preprocessing script also creates visual outputs to make the labels easier to inspect:

- Original pre-disaster image
- Original post-disaster image
- Polygon overlay on the post-disaster image
- Grayscale segmentation mask
- Color segmentation mask
- Damage class statistics CSV
- Building bounding boxes CSV

Example outputs are saved in:

```text
results/week1/visualizations/
```

### Why Week 1 Matters

Week 1 verifies that the raw dataset is being read correctly. In segmentation projects, this step is very important because a model can only learn correctly if the masks match the images. The visual overlays and masks help confirm that building polygons are aligned with the satellite images.

## Week 2: Baseline Segmentation Pipeline

Week 2 turned the preprocessing work into the first PyTorch training pipeline.

The main scripts for this stage are:

```text
src/week2_dataset.py
src/week2_model.py
src/week2_train_baseline.py
```

### Dataset Pipeline

The dataset class loads each sample as a paired pre/post image. The two RGB images are concatenated into one 6-channel tensor:

```text
pre RGB + post RGB = 6 channels
```

For binary segmentation, the damage mask is converted into:

```text
building pixel = 1
background pixel = 0
```

The dataset pipeline also creates deterministic train, validation, and test split files:

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

### Week 2 Data Augmentation

Albumentations is used for resizing, normalization, tensor conversion, and training augmentation. The pipeline includes:

- Resize
- Horizontal flip
- Vertical flip
- Random 90-degree rotation
- Shift, scale, and rotation
- Brightness and contrast adjustment

These augmentations are useful because satellite images can vary in orientation, lighting, resolution, and image quality.

The shared dataset code supports Week 3 augmentation too, but `src/week2_train_baseline.py` explicitly runs with the Week 2 augmentation setting so the original baseline stays comparable.

### Week 2 Baseline Model

The Week 2 model is the first compact U-Net baseline implemented in:

```text
src/week2_model.py
```

U-Net is a standard architecture for image segmentation. It uses an encoder-decoder structure:

- The encoder extracts spatial features from the input image.
- The decoder upsamples features back to mask resolution.
- Skip connections help preserve fine spatial details such as building boundaries.

The model input has 6 channels and the output has 1 channel for binary segmentation logits. This model is intentionally kept as the simple baseline so later experiments can be compared against it.

### Initial Training

The Week 2 training script trains the first U-Net baseline and saves the best checkpoint:

```text
src/week2_train_baseline.py
results/week2/checkpoints/week2_unet_binary_best.pt
```

This baseline provided the first end-to-end test of:

- Image loading
- Mask generation
- Batch creation
- Model forward pass
- Loss computation
- Backpropagation
- Validation Dice score
- Checkpoint saving

The Week 2 training setup uses `BCEWithLogitsLoss`, reports loss and Dice, and defaults to a short training run. This stage confirmed that the project had a working segmentation training loop.

## Week 3: Cleaning, Debugging, and Reliability

Week 3 does not overwrite the Week 2 baseline. Instead, it creates a separate improved baseline so the project can compare the original model against the upgraded system.

The main additions and updates are:

```text
src/week3_dataset_statistics.py
src/week3_model.py
src/week3_train.py
src/week2_dataset.py
```

`week2_model.py` and `week2_train_baseline.py` remain the Week 2 baseline. `week3_model.py` and `week3_train.py` are the improved Week 3 model path.

### Model Separation

| Stage | Model file | Training file | Checkpoint |
|---|---|---|---|
| Week 2 baseline | `src/week2_model.py` | `src/week2_train_baseline.py` | `results/week2/checkpoints/week2_unet_binary_best.pt` |
| Week 3 improved baseline | `src/week3_model.py` | `src/week3_train.py` | `results/week3/checkpoints/week3_unet_binary_best.pt` |

The two model files currently share the same compact U-Net architecture, but they are intentionally separated. This keeps the Week 2 baseline stable while allowing Week 3 to add improved training, metrics, debugging tools, and future architecture changes without changing the original experiment.

## Dataset Cleaning

The most important Week 3 task was dataset cleaning.

Invalid or noisy samples are skipped before training. The dataset now filters out samples when:

- The post-disaster image or label file is missing
- The label JSON cannot be loaded
- No valid damage class exists
- Polygons cannot be parsed
- The generated mask is empty

The project also ignores `un-classified` and unknown labels. This avoids training the model on ambiguous or unreliable annotations.

### Dataset Statistics

A new statistics script was added:

```text
src/week3_dataset_statistics.py
```

It saves report-friendly CSV files to:

```text
results/week3/metrics/
```

Current training dataset statistics:

| Metric | Value |
|---|---:|
| total samples | 2799 |
| valid samples | 2241 |
| skipped samples | 558 |
| empty masks | 0 |
| missing files | 0 |
| invalid label json | 0 |
| invalid polygons | 0 |
| no valid damage class | 558 |
| ignored unclassified/unknown buildings | 2993 |
| buildings no-damage | 117426 |
| buildings minor-damage | 14980 |
| buildings major-damage | 14161 |
| buildings destroyed | 13227 |

These statistics are useful for the research/report section because they describe the quality and class distribution of the training data.

## Week 3 Improved Loss Function

The Week 3 training script now uses a combined BCE + Dice loss:

```text
loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

BCE helps the model classify individual pixels correctly. Dice loss helps optimize overlap between the predicted mask and the ground truth mask. This combination is common in segmentation because it handles both pixel-level accuracy and mask-level shape quality.

## Week 3 Better Augmentation

Week 3 keeps the Week 2 geometric and brightness augmentations, then adds blur/noise augmentation in the shared dataset pipeline. This makes the improved baseline more robust to satellite image quality differences while keeping the Week 2 runner able to use the original augmentation setup.

## Week 3 Better Metrics

The Week 3 validation loop reports multiple segmentation metrics:

- Dice
- IoU
- Precision
- Recall
- F1-score

Dice and F1 are equivalent in this binary segmentation setup. IoU gives a stricter overlap measure. Precision shows how many predicted building pixels are correct, while recall shows how many true building pixels were found.

## Week 3 Prediction Visualization

The Week 3 training script saves prediction examples whenever validation Dice improves.

Prediction outputs are saved in:

```text
results/week3/predictions/
```

For each sample, the script saves:

- Input post-disaster image
- Ground truth mask
- Predicted mask
- Combined input/ground-truth/prediction panel

This is important because numerical metrics do not always tell the full story. Visualizations help detect problems such as shifted masks, empty predictions, noisy predictions, or predictions that look reasonable despite low scores.

## Overfit Test

Week 3 also adds an overfit test mode:

```text
python src/week3_train.py --overfit-samples 8 --epochs 50 --batch-size 2 --small-model
```

This trains and validates on the same small set of clean samples. The model should eventually reach a very high Dice score, ideally above 0.9.

If the model cannot overfit 5-10 samples, that usually means there is a pipeline issue such as:

- Incorrect masks
- Bad image/mask alignment
- Wrong loss function
- Broken model output shape
- Label filtering problems
- Too much augmentation during debugging

In this project, the overfit mode disables training augmentation by using validation transforms for the tiny subset. This makes the test easier to interpret.

## Output Folder Structure

The project now keeps experiment outcomes under one results tree:

```text
results/
+-- week1/
|   \-- visualizations/
+-- week2/
|   +-- checkpoints/
|   \-- predictions/
+-- week3/
|   +-- checkpoints/
|   +-- config/
|   +-- metrics/
|   +-- predictions/
|   \-- visualizations/
+-- week4/
|   +-- checkpoints/
|   +-- config/
|   +-- metrics/
|   +-- predictions/
|   \-- visualizations/
+-- week5/
|   +-- checkpoints/
|   +-- config/
|   +-- confusion_matrices/
|   +-- metrics/
|   +-- predictions/
|   \-- visualizations/
+-- week6/
|   +-- experiment_baseline/
|   +-- experiment_attention_unet/
|   +-- experiment_resnet50/
|   +-- hyperparameter_optimization/
|   \-- comparative summaries
+-- week7/
|   +-- experiment_siamese_concat/
|   +-- experiment_siamese_difference/
|   +-- experiment_siamese_cbam/
|   \-- experiment_siamese_nonlocal/
+-- week8/
|   +-- class_distribution_train_val.csv
|   +-- per_disaster_class_distribution.csv
|   +-- selected_extra_minority_samples.csv
|   \-- balanced-data audit files
+-- week9/
|   +-- experiment_multitask_difference/
|   +-- experiment_multitask_cbam_difference/
|   \-- loss-balancing ablations
\-- week10/
    +-- experiment_week10a_building_masked_damage_loss/
    \-- experiment_week10a1_soft_masked_damage_loss/
```

Each folder has a clear purpose:

| Folder | Purpose |
|---|---|
| `results/week1/visualizations/` | Week 1 sample visualizations, masks, overlays, and CSV summaries |
| `results/week2/checkpoints/` | Week 2 baseline model weights |
| `results/week2/predictions/` | Week 2 qualitative prediction outputs, when generated |
| `results/week3/` | Week 3 metrics, config, checkpoints, prediction panels, visualizations, and failure analysis |
| `results/week4/` | Week 4 pretrained-encoder metrics, config, checkpoints, prediction panels, visualizations, and failure analysis |
| `results/week5/` | Week 5 multiclass metrics, confusion matrices, color-mask predictions, checkpoints, and failure analysis |
| `results/week6/` | Isolated research experiment folders, ablation summaries, HPO studies, TensorBoard logs, precision-recall curves, and qualitative panels |
| `results/week7/` | Temporal Siamese experiment folders, fusion/attention comparisons, temporal difference outputs, and attention-oriented analysis |
| `results/week8/` | Class-distribution audits, per-disaster class summaries, selected minority sample lists, and balanced split reports |
| `results/week9/` | Multi-task Siamese experiment outputs, auxiliary building metrics, damage metrics, checkpoints, and lambda ablation summaries |
| `results/week10/` | Building-masked and soft building-weighted damage-loss experiments using the Week 9 architecture |

## How to Run the Project

Create dataset statistics:

```powershell
python src\week3_dataset_statistics.py --data-dir data --split train --output-dir results\week3\metrics
```

Train the Week 2 baseline:

```powershell
python src\week2_train_baseline.py --epochs 5 --batch-size 4 --image-size 512
```

Train the Week 3 improved baseline:

```powershell
python src\week3_train.py --epochs 20 --batch-size 4 --image-size 512
```

Train the Week 4 ResNet34 encoder U-Net:

```powershell
python src\week4_train.py --epochs 20 --batch-size 4 --image-size 512
```

Train the Week 5 multiclass damage model:

```powershell
python src\week5_train.py --epochs 20 --batch-size 4 --image-size 512
```

Create the Week 6 research result tree:

```powershell
python src\week6\week6_experiment_runner.py --scaffold-only
```

Run the Week 6 Morocco-focused research suite:

```powershell
.\run_week6_experiments.ps1
```

Run a Week 7 temporal Siamese baseline:

```powershell
python src\week7\week7_experiment_runner.py --experiment siamese_difference --morocco-adaptation --epochs 20 --batch-size 4
```

Run the Week 8 class-imbalance audit:

```powershell
python src\week8\week8_class_distribution.py --data-dir data --split-dir splits --output-dir results\week8 --morocco-adaptation
```

Run the Week 9 multi-task Siamese model:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --morocco-adaptation --epochs 50 --batch-size 4
```

Run the Week 10 building-masked damage-loss experiment:

```powershell
python src\week10\week10_train_masked_loss.py --experiment multitask_cbam_difference --morocco-adaptation --epochs 50 --batch-size 4
```

Run the overfit debugging test:

```powershell
python src\week3_train.py --overfit-samples 8 --epochs 50 --batch-size 2 --small-model
```

## Current Project Status

The project now has a complete baseline segmentation system:

- Raw xBD labels are parsed correctly.
- Building polygons are converted into segmentation masks.
- Dataset splits are created and loaded with PyTorch.
- Invalid and noisy samples are filtered.
- A Week 2 U-Net baseline is available.
- A separate Week 3 improved U-Net baseline is available.
- Week 3 training uses BCE + Dice loss.
- Week 3 validation reports multiple metrics.
- Week 2 and Week 3 checkpoints are saved separately.
- Week 3 prediction examples are saved for visual inspection.
- Dataset statistics are available for the report.
- A Week 4 ResNet34 encoder U-Net is available as the next model upgrade.
- A Week 5 multiclass damage segmentation model is available for severity prediction.
- Week 6 provides a reusable research experiment framework for ablations, HPO, samplers, advanced losses, and model comparisons.
- Week 7 provides temporal Siamese models that process pre-disaster and post-disaster images as separate streams before feature fusion.
- Week 8 provides rare-class auditing and targeted sample selection for improving minor/major/destroyed learning.
- Week 9 provides a multi-task Siamese architecture with auxiliary building-mask heads and a damage head.
- Week 10 provides building-aware damage-loss experiments that reduce background dominance during training.
- An overfit test is available for debugging the full pipeline.

## Week 4 Model

The highest-impact, lowest-risk upgrade is to keep the U-Net segmentation structure but replace the scratch encoder with a pretrained ResNet34 encoder.

### Why This Upgrade Wins

The current U-Net learns visual features from scratch. That means part of its capacity is spent relearning general image patterns such as edges, textures, corners, roof shapes, and object boundaries.

A pretrained ResNet34 encoder already contains strong low-level and mid-level visual features learned from large image datasets. Reusing those features should give the segmentation model:

- Stronger edge and boundary detection
- Better texture awareness
- More useful geometry priors
- Faster convergence
- More stable training
- Less sensitivity to noisy or limited labels

### Why It Matters for xBD

xBD satellite images contain visual patterns that benefit directly from pretrained convolutional features:

- Roof structures
- Roads and building boundaries
- Rubble and debris textures
- Flood and water patterns
- Fire and disaster-scene texture changes

These are exactly the kinds of visual primitives that a ResNet encoder can reuse before the decoder learns the project-specific building mask output.

### Expected Impact

Compared with the Week 3 U-Net baseline, a realistic improvement target is:

| Metric | Expected Change |
|---|---:|
| Dice | +0.05 to +0.12 |
| Convergence speed | 2x to 4x faster |
| Training stability | Significantly better |

The implementation preserves the current Week 3 results as the comparison baseline and saves the new experiment separately under:

```text
results/week4/
```

## Week 5 Model

Week 5 moves from binary building segmentation to multiclass disaster damage segmentation. The task changes from:

```text
Where are buildings?
```

to:

```text
How damaged are the buildings?
```

This is the core scientific objective of the xBD dataset.

### Week 5 Classes

The model predicts five classes:

| Class ID | Meaning |
|---:|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

The output tensor changes from the Week 4 binary shape:

```text
[1, H, W]
```

to the Week 5 multiclass shape:

```text
[5, H, W]
```

### Architecture

Week 5 keeps the ResNet34 encoder U-Net architecture from Week 4 but changes the segmentation head to output five channels. The training dataset runs with:

```text
target_mode="multiclass"
```

so targets are class-id masks with shape:

```text
[H, W]
```

and dtype:

```text
torch.long
```

### Loss Function

Binary BCE is no longer correct for this task. Week 5 uses:

```text
CrossEntropyLoss + multiclass Dice loss
```

CrossEntropy handles class prediction, while Dice improves overlap quality for each class.

### Class Imbalance

The xBD damage labels are highly imbalanced. No-damage buildings are much more common than damaged buildings:

| Class | Building Count |
|---|---:|
| No damage | 117426 |
| Minor damage | 14980 |
| Major damage | 14161 |
| Destroyed | 13227 |

Week 5 therefore supports weighted CrossEntropy. The default class weights are:

```text
[1.0, 0.2, 1.5, 1.7, 1.8]
```

This reduces the tendency to predict mostly no-damage.

### Metrics and Visualization

Week 5 tracks multiclass metrics:

- Per-class Dice
- Mean foreground Dice
- Per-class IoU
- Mean IoU
- Pixel accuracy
- Macro F1
- Confusion matrix

The most important scientific metrics are `major_damage` Dice and `destroyed` Dice because these classes are most relevant for disaster response.

Qualitative outputs now use color masks:

| Class | Color |
|---|---|
| Background | Black |
| No damage | Green |
| Minor damage | Yellow |
| Major damage | Orange |
| Destroyed | Red |

Each saved sample includes the input image, ground-truth color mask, predicted color mask, and prediction overlay.

Confusion matrices are saved under:

```text
results/week5/confusion_matrices/
```

These are useful for studying common errors such as minor-to-major confusion or major-to-destroyed confusion.

## Week 6: Research Experiment Framework and Architecture Ablations

Week 6 turns the Week 5 multiclass baseline into a more complete research platform. Instead of training one model at a time with separate scripts, Week 6 introduces an experiment runner that can build different models, losses, samplers, schedulers, metrics, visualizations, and output folders from one controlled interface.

The main Week 6 files are:

```text
src/week6/week6_experiment_runner.py
src/week6/week6_model_attention_unet.py
src/week6/week6_model_unetplusplus.py
src/week6/week6_model_resnet50_unet.py
src/week6/week6_model_deeplabv3plus.py
src/week6/week6_losses.py
src/week6/week6_sampler.py
src/week6/week6_metrics.py
src/week6/week6_visualization.py
src/week6/week6_scheduler.py
src/week6/week6_analysis.py
src/week6/week6_utils.py
```

### Experiment Runner Architecture

The central controller is `week6_experiment_runner.py`. It receives command-line settings, creates isolated result folders, builds the selected model, prepares train/validation dataloaders, trains the model, evaluates metrics, saves checkpoints, and writes summary artifacts.

The runner is organized around a repeatable experiment lifecycle:

1. Read split files and optionally filter sample IDs for Morocco-relevant disasters such as earthquake, flood, flooding, wildfire, and fire.
2. Build dataloaders using the shared xBD dataset pipeline and Week 6 augmentations.
3. Build the selected architecture through `build_model`.
4. Build the selected loss through `build_loss`.
5. Optionally build class-aware sampling through `build_weighted_sampler`.
6. Train and validate for each epoch.
7. Save the best checkpoint, metrics CSV files, per-class metrics, confusion matrices, precision-recall rows, and qualitative panels.
8. Optionally run multi-seed, k-fold, or Optuna hyperparameter optimization workflows.

This design matters because later experiments can be compared fairly. Each run has its own folder, configuration file, metrics history, and visual outputs, so a new experiment does not overwrite older baselines.

### Week 6 Input and Target Format

Week 6 continues to use paired pre/post images as a 6-channel tensor:

```text
[pre_R, pre_G, pre_B, post_R, post_G, post_B]
```

The target remains a multiclass damage mask:

```text
[H, W] with class IDs 0-4
```

The output of every Week 6 model is:

```text
[batch_size, 5, H, W]
```

where the five channels represent background, no damage, minor damage, major damage, and destroyed.

### Attention U-Net

The scratch Attention U-Net is implemented in:

```text
src/week6/week6_model_attention_unet.py
```

Its architecture is:

- A 6-channel input convolution block
- Four encoder levels with max-pooling
- A bottleneck convolution block
- Four decoder levels with transposed-convolution upsampling
- Attention gates on skip connections
- A final 1x1 convolution that maps decoder features to five damage classes

The attention gate receives two feature maps: the decoder gate feature and the matching encoder skip feature. Both are projected with 1x1 convolutions, added together, passed through ReLU, reduced to a single attention map, and passed through sigmoid. The skip feature is multiplied by this attention map before it is concatenated with the decoder feature.

This means the decoder does not receive every skip feature equally. It learns to emphasize spatial regions that are useful for damage prediction and reduce irrelevant background detail.

### UNet++ Architecture

UNet++ is implemented in:

```text
src/week6/week6_model_unetplusplus.py
```

The key idea is nested skip connections. A normal U-Net has one skip path from each encoder level to each decoder level. UNet++ creates intermediate decoder nodes such as `x0_1`, `x0_2`, `x0_3`, and `x0_4`, where each node combines features from multiple earlier nodes and an upsampled deeper feature.

This gives the model a denser multi-scale feature path. For satellite damage segmentation, this is useful because:

- Tiny buildings need high-resolution detail.
- Large building blocks need broader context.
- Damage boundaries may be subtle and spatially fragmented.
- Nested skip paths help bridge the semantic gap between shallow texture features and deep disaster-context features.

The final prediction comes from the deepest nested decoder node `x0_4`, then a 1x1 head maps it to five classes.

### ResNet50 U-Net

The Week 6 ResNet50 U-Net is implemented in:

```text
src/week6/week6_model_resnet50_unet.py
```

This model upgrades the Week 4 ResNet34 idea to a deeper ResNet50 encoder. Since ResNet50 normally expects a 3-channel RGB image, the first convolution is adapted to accept the 6-channel pre/post stack. The encoder then produces a feature pyramid:

| Feature level | Channels | Purpose |
|---|---:|---|
| stem | 64 | Low-level edges, roofs, roads, and texture |
| encoder1 | 256 | Local building parts |
| encoder2 | 512 | Mid-level object patterns |
| encoder3 | 1024 | Larger spatial and disaster context |
| encoder4 | 2048 | Deep semantic representation |

The decoder uses the `DecoderBlock` pattern from the earlier ResNet U-Net code. It upsamples deep features and combines them with matching skip features:

```text
enc4 -> decoder4 + enc3
decoder4 -> decoder3 + enc2
decoder3 -> decoder2 + enc1
decoder2 -> decoder1 + stem
decoder1 -> final upsample -> 5-class head
```

The pretrained encoder provides stronger visual features than the scratch Week 5 model, while the U-Net decoder restores pixel-level resolution for segmentation.

### DeepLabV3-Based Experiment

Week 6 also includes a DeepLab-style experiment wrapper:

```text
src/week6/week6_model_deeplabv3plus.py
```

The filename is kept for continuity, while the saved metadata uses the correct DeepLabV3 architecture name. DeepLab is useful as a comparison point because it uses atrous convolution and broader context aggregation rather than the pure U-Net skip-decoder design. This helps test whether disaster damage segmentation benefits more from dense local boundaries or larger contextual receptive fields.

### Loss Functions

Week 6 introduces a loss factory in:

```text
src/week6/week6_losses.py
```

Supported losses include:

- CrossEntropy + Dice
- Focal loss
- Tversky loss
- Focal Tversky loss
- Combined loss wrappers

These losses address class imbalance in different ways. CrossEntropy provides stable class supervision. Dice improves region overlap. Focal loss emphasizes difficult pixels. Tversky-style losses allow false positives and false negatives to be weighted differently, which is useful when rare damage classes are missed.

### Samplers, Schedulers, and Metrics

`week6_sampler.py` adds class-aware sampling so batches can contain more images with rare damage classes. This is important because minor and major damage appear far less often than no-damage buildings.

`week6_scheduler.py` supports learning-rate strategies such as ReduceLROnPlateau and warmup cosine scheduling. This makes longer research runs more stable than a fixed learning rate.

`week6_metrics.py` expands evaluation beyond one mean score. It computes confusion matrices, per-class Dice, IoU, boundary IoU, mean IoU, and precision-recall points. Boundary IoU is especially useful for segmentation because two masks can have similar class counts but poor building outlines.

### Visualization and Analysis

`week6_visualization.py` saves qualitative outputs such as:

- Prediction panels
- Color masks
- Overlays
- Confidence maps
- Error heatmaps

This makes Week 6 a research-quality experiment layer, not only a training script. The user can inspect whether improvements are real or only metric artifacts.

## Week 7: Temporal Siamese Damage Segmentation

Week 7 changes the way the model sees time. Earlier models stack pre-disaster and post-disaster RGB images into one 6-channel tensor at the input. This is simple, but it forces the first convolution layer to learn temporal comparison immediately.

Week 7 separates the two images into two temporal streams:

```text
pre-disaster RGB image  -> pre encoder
post-disaster RGB image -> post encoder
```

Then it fuses features at multiple semantic levels before decoding a damage mask.

The main Week 7 files are:

```text
src/week7/week7_dataset.py
src/week7/week7_model_siamese_resnet50_unet.py
src/week7/week7_temporal_fusion.py
src/week7/week7_attention.py
src/week7/week7_experiment_runner.py
src/week7/week7_losses.py
src/week7/week7_metrics.py
src/week7/week7_visualization.py
src/week7/week7_inference.py
```

### Dataset Interface

`XBDTemporalDamageDataset` wraps the existing xBD dataset and splits the 6-channel tensor into two RGB tensors:

```text
image[:3]   -> pre_image
image[3:6] -> post_image
```

The returned batch contains:

```text
pre_image
post_image
image
mask
sample_id
```

This keeps compatibility with the older dataset pipeline while giving temporal models a cleaner input format.

### Siamese ResNet50 Encoder

The core Week 7 model is:

```text
src/week7/week7_model_siamese_resnet50_unet.py
```

Each RGB image is processed by a ResNet50 encoder that returns a feature pyramid:

```text
stem, enc1, enc2, enc3, enc4
```

The model can use separate pre/post encoders or share the same encoder weights. Separate encoders allow each temporal branch to specialize, while shared encoders force both images into the same feature space and reduce parameters.

### Temporal Fusion

Temporal fusion is implemented in:

```text
src/week7/week7_temporal_fusion.py
```

The fusion module compares pre-disaster and post-disaster features at each encoder level. It supports:

| Strategy | Description |
|---|---|
| concat | Concatenates pre and post features |
| difference | Uses absolute feature difference |
| concat_difference | Concatenates pre, post, and absolute difference |
| gated_fusion | Learns a sigmoid gate that blends pre and post features per pixel |

After fusion, a 1x1 projection maps the fused tensor back to the original channel count. This is important because the decoder expects the same channel dimensions regardless of fusion strategy.

The difference strategy is especially meaningful for damage detection because damage is a change over time. It forces the model to focus on what changed between the pre and post images rather than treating both images as unrelated channels.

### Attention Modules

Week 7 introduces bottleneck attention modules in:

```text
src/week7/week7_attention.py
```

Available attention types include:

- Identity attention for baseline comparison
- Squeeze-and-excitation channel attention
- CBAM channel + spatial attention
- Non-local self-attention

CBAM first estimates which channels are important, then estimates which spatial regions are important. Non-local attention models long-range relationships across the feature map. In disaster imagery, this can help because damaged neighborhoods, flood regions, and fire-affected areas often have spatial structure larger than one building.

### Week 7 Decoder and Output

After temporal fusion, the decoder follows a U-Net pattern:

```text
fused enc4 -> attention -> decoder4 with fused enc3
decoder4  -> decoder3 with fused enc2
decoder3  -> decoder2 with fused enc1
decoder2  -> decoder1 with fused stem
decoder1  -> final upsample -> 5-class damage head
```

The output is still:

```text
[batch_size, 5, H, W]
```

but the internal representation is now explicitly temporal. This is a major architectural step because the model no longer has to discover temporal comparison only from the first convolution.

## Week 8: Rare-Class Imbalance Audit and Targeted Data Expansion

Week 8 is not primarily a neural-network architecture week. It is a data architecture week. The problem discovered after multiclass training is that rare damage classes, especially `minor_damage`, are much harder to learn than background, no-damage, and destroyed.

The main Week 8 files are:

```text
src/week8/week8_class_distribution.py
src/week8/week8_select_minority_samples.py
src/week8/week8_prepare_balanced_data.py
src/week8/week8_prune_dataset.py
```

### Class-Distribution Audit

`week8_class_distribution.py` reads train/validation split files, loads post-disaster labels, rasterizes WKT polygons into class-id masks, and counts:

- Pixels per class
- Images containing each class
- Building polygons per class
- Per-disaster class distributions

The script groups samples by disaster keywords such as earthquake, flood, flooding, wildfire, and fire. This is useful because the project is oriented toward Morocco-relevant disaster types, and the class imbalance may differ by disaster family.

### Data Selection Architecture

`week8_select_minority_samples.py` scans candidate xBD data and ranks samples by rare-class usefulness. Instead of blindly adding more images, the project selects samples that actually contain useful minority damage classes.

The selection logic checks:

- Whether the candidate has matching pre image, post image, and label JSON
- Whether the sample belongs to the desired disaster scope
- Whether the label contains valid damage classes
- Whether minority-class buildings and pixels are present
- Whether the sample is useful enough to keep

The output is:

```text
results/week8/selected_extra_minority_samples.csv
```

This file becomes a controlled bridge between raw downloaded xBD data and the training set.

### Balanced Training Split

`week8_prepare_balanced_data.py` copies selected extra samples into a separate folder:

```text
data/week8_extra/
```

and creates:

```text
splits/week8_train_balanced.txt
```

The original train, validation, and test split files are not modified. This is an important experimental design choice because validation and test results stay comparable. Week 8 expands only the training data.

### Pruning Support

`week8_prune_dataset.py` identifies non-Morocco-scope or invalid samples that can be removed to save storage. It writes deletion candidates first and only deletes files when explicit confirmation flags are provided.

Week 8 therefore acts as a controlled data-governance layer: audit, select, copy, compare, and preserve fair evaluation.

## Week 9: Multi-Task Siamese Damage Segmentation

Week 9 introduces the most complete model architecture in the project. The model predicts three related outputs at once:

```text
1. Pre-disaster building mask
2. Post-disaster building mask
3. Multiclass damage mask
```

The main Week 9 files are:

```text
src/week9/week9_dataset.py
src/week9/week9_model_multitask_siamese.py
src/week9/week9_losses.py
src/week9/week9_metrics.py
src/week9/week9_train_multitask.py
```

### Why Multi-Task Learning Is Used

Damage classification depends on knowing where buildings are. Background pixels do not have meaningful damage states. Earlier models had to learn building localization and damage severity from only one damage mask. Week 9 makes this dependency explicit by adding auxiliary building-mask tasks.

The intended learning signal is:

- The pre-building head teaches the model where buildings existed before the disaster.
- The post-building head teaches the model where visible buildings or building remains appear after the disaster.
- The damage head learns severity using a feature representation already shaped by building supervision.

This is especially helpful for rare classes because the model receives dense building-boundary supervision even when rare damage pixels are scarce.

### Dataset Output

`XBDMultiTaskSampleDataset` returns:

```text
pre_image
post_image
pre_building_mask
post_building_mask
damage_mask
sample_id
```

The building masks are binary class-id masks, while the damage mask is the five-class xBD target. `XBDMultiTaskCombinedDataset` can combine the original training set with selected Week 8 extra samples.

### Multi-Task Siamese Architecture

The model is implemented in:

```text
src/week9/week9_model_multitask_siamese.py
```

It uses a shared ResNet50 encoder for both temporal images:

```text
pre_image  -> shared ResNet50 encoder
post_image -> shared ResNet50 encoder
```

The shared encoder produces feature pyramids for both images:

```text
stem, enc1, enc2, enc3, enc4
```

Each corresponding feature level is fused through `TemporalFusion`. By default, Week 9 uses difference fusion with CBAM attention:

```text
fused[level] = TemporalFusion(pre[level], post[level])
```

The fused deep feature `enc4` is passed through bottleneck attention, then the U-Net decoder reconstructs high-resolution segmentation features:

```text
fused enc4 -> CBAM -> decoder4 + fused enc3
decoder4  -> decoder3 + fused enc2
decoder3  -> decoder2 + fused enc1
decoder2  -> decoder1 + fused stem
decoder1  -> final upsample
```

The final feature map is shared by three heads:

| Head | Output channels | Meaning |
|---|---:|---|
| pre_building_head | 2 | background vs pre-disaster building |
| post_building_head | 2 | background vs post-disaster building |
| damage_head | 5 | background, no damage, minor, major, destroyed |

The forward pass returns a dictionary:

```text
pre_building_logits
post_building_logits
damage_logits
```

This architecture keeps one shared temporal representation but lets each task have its own final classifier.

### Multi-Task Loss

`MultiTaskDamageLoss` combines three losses:

```text
total_loss =
    lambda_pre * pre_building_loss
  + lambda_post * post_building_loss
  + lambda_damage * damage_loss
```

The default intent is to give damage prediction the largest weight:

```text
lambda_pre = 1
lambda_post = 1
lambda_damage = 3
```

The building losses use binary CrossEntropy-style class prediction through two-logit CrossEntropy plus foreground Dice. The damage loss can use the Week 6 loss factory, including CrossEntropy + Dice or rare-class weighted variants.

### Optimizer Architecture

Week 9 uses parameter groups:

- Encoder learning rate
- Fusion learning rate
- Decoder/head learning rate

This is useful because pretrained encoder features often need smaller updates, while newly initialized fusion, decoder, and output heads need faster learning.

### Week 9 Metrics

The metric layer reports building-task and damage-task behavior separately. This matters because a model may segment buildings well but still confuse minor, major, and destroyed classes. Week 9 also supports damage confusion excluding background so the analysis focuses on mistakes between building damage classes.

## Week 10: Building-Aware Damage Losses

Week 10 keeps the Week 9 multi-task Siamese CBAM architecture fixed and changes the damage loss. This isolates the experiment: if performance changes, the difference comes from the objective function rather than a new model.

The main Week 10 files are:

```text
src/week10/week10_train_masked_loss.py
src/week10/week10_train_soft_masked_loss.py
src/week10/week10_train_class_weighted_ce.py
src/week9/week9_losses.py
```

### Motivation

Damage labels are meaningful mainly on building pixels. In a full satellite image, most pixels are background: roads, fields, water, empty ground, and vegetation. If loss is computed equally over all pixels, the background can dominate optimization.

Week 10 tests whether damage learning improves when the loss pays more attention to building pixels.

### Week 10A: Hard Building-Masked Damage Loss

`week10_train_masked_loss.py` sets:

```text
damage_loss = "building_masked_ce_dice"
```

This uses `BuildingMaskedDamageCEDiceLoss`. The loss creates a valid-pixel mask:

```text
valid = targets > 0
```

Only foreground building pixels contribute to the damage CrossEntropy and Dice terms. If a batch has no building pixels, the damage loss returns zero for that batch.

The advantage is that the objective focuses on classifying building damage severity. The risk is that the model receives much less background supervision for the damage head, so false positives outside buildings must be monitored carefully.

### Week 10A.1: Soft Building-Weighted Damage Loss

`week10_train_soft_masked_loss.py` sets:

```text
damage_loss = "soft_building_weighted_ce_dice"
```

This uses `SoftBuildingWeightedDamageCEDiceLoss`. Instead of removing background pixels entirely, it gives pixels different weights:

```text
background pixels: 0.2
building pixels:   1.0
```

This keeps background supervision in the objective while still emphasizing the pixels where damage labels matter most. It is a compromise between ordinary full-image loss and hard foreground-only loss.

### Week 10A.1 Results Interpretation

The soft building-weighted loss was more stable than the hard building-masked loss, but it did not outperform the earlier plain full-image CrossEntropy damage baseline overall.

The earlier baseline used:

```text
loss_damage = CrossEntropy(damage_logits, damage_targets)
```

Against that baseline, the soft building-weighted experiment showed a small improvement in building-only damage Dice and rare-class recall, but it reduced the main global damage metrics and weakened the auxiliary building segmentation heads.

| Metric | Plain CE baseline | Soft building-weighted loss |
|---|---:|---:|
| best epoch | 32 | 26 |
| val damage mean Dice | 0.5399 | 0.4592 |
| val building-only damage mean Dice | 0.6298 | 0.6427 |
| val pre-building Dice | 0.7667 | 0.7267 |
| val post-building Dice | 0.7650 | 0.7283 |
| val damage pixel accuracy | 0.9574 | 0.9186 |
| val rare-class recall | 0.4896 | 0.5025 |
| val minor-damage Dice | 0.0464 | 0.0588 |
| val major-damage Dice | 0.5543 | 0.4440 |
| val destroyed Dice | 0.7834 | 0.6655 |

This means the soft weighting experiment was useful diagnostically, but it was not a better default objective. The small gains in building-only damage Dice, rare-class recall, and minor-damage Dice were not large enough to justify the losses in global damage Dice, building Dice, major-damage Dice, destroyed Dice, and pixel accuracy.

The likely explanation is that background pixels are not merely easy negatives. They help the model learn global context, non-building suppression, and building boundaries. Reducing their contribution too aggressively, even with a soft weight of 0.2, weakened the shared representation used by both the building heads and the damage head.

The main remaining failure mode is therefore not background dominance. It is minor-damage separability. Both the plain CE baseline and the soft building-weighted model perform poorly on minor damage:

| Metric | Plain CE baseline | Soft building-weighted loss |
|---|---:|---:|
| val damage Dice, minor damage | 0.0464 | 0.0588 |
| val building-only Dice, minor damage | 0.0518 | 0.0733 |

The soft loss improves minor damage slightly, but the absolute score remains very low. Future work should treat plain CE as the stronger baseline and focus on targeted minor-damage improvements such as moderate class weighting, focal loss with a low gamma, rare-class oversampling, or label-quality inspection for visually ambiguous minor-damage cases.

### Week 10B: Class-Weighted Full-Image Damage Loss

Week 10B keeps the lesson from Week 10A.1: background supervision should remain active. Instead of masking or strongly down-weighting background pixels, Week 10B returns to full-image damage loss and uses class weights to focus specifically on the weak class, minor damage.

The new runner is:

```text
src/week10/week10_train_class_weighted_ce.py
```

The highest-priority experiment is class-weighted CrossEntropy:

```text
damage_loss = "weighted_cross_entropy"
```

The trial weight sets are:

| Trial | background | no damage | minor damage | major damage | destroyed |
|---|---:|---:|---:|---:|---:|
| A, conservative | 1.0 | 1.0 | 2.5 | 1.5 | 1.2 |
| B, stronger minority push | 1.0 | 1.0 | 4.0 | 1.5 | 1.2 |
| C, aggressive | 1.0 | 1.0 | 6.0 | 2.0 | 1.5 |

Trial A should run first. Trial B should run only if Trial A improves minor damage without a large drop in global damage Dice. Trial C should be reserved for cases where Trial B clearly improves minor damage and the model still has enough global stability.

The recommended Week 10B runs use the normal shuffled dataloader rather than the class-aware weighted sampler. This isolates the effect of the loss weights from the effect of sample oversampling.

If weighted CE improves the minor-damage metrics, Week 10B also supports weighted CE plus multiclass Dice:

```text
damage_loss = "weighted_cross_entropy_dice"
loss = 0.7 * weighted_ce + 0.3 * multiclass_dice
```

This combines CE's pixel classification signal with Dice's overlap consistency. The Dice term is multiclass Dice and ignores background by default.

If weighted CE plateaus, Week 10B can switch to focal CE:

```text
damage_loss = "focal"
gamma = 2.0
alpha = selected trial class weights
```

The most important monitoring metrics for Week 10B are minor-damage Dice, minor-damage IoU, and rare-class recall. Overall damage mean Dice, destroyed Dice, and the pre/post building Dice scores should be monitored as stability checks.

### Week 10 Relationship to Week 9

Week 10 reuses:

- The Week 9 dataset
- The Week 9 multi-task Siamese model
- The Week 9 training loop
- The Week 9 optimizer groups
- The Week 9 metrics

Only the damage-loss function and result root are changed. This makes Week 10 a clean ablation study on loss design.

## Week 11: Object-Level Disaster Damage Classification

Week 11 changes the scientific direction of the project from dense segmentation toward building-level damage assessment. The segmentation work is treated as sufficiently mature, and architecture experimentation is frozen around the strongest design family:

```text
Siamese ResNet50-UNet
difference fusion
CBAM
soft building-aware weighting or plain CE/focal CE, depending on the best final run
```

The purpose of Week 11 is to convert the xBD supervision from millions of noisy pixels into one sample per building. This creates a cleaner learning problem for damage severity because the classifier can reason over a complete object crop rather than isolated pixels.

### Building Instance Extraction

The new extraction script is:

```text
src/week11/week11_extract_buildings.py
```

It reads the existing xBD split files and ground-truth polygon labels. For each building polygon, it rasterizes a binary mask, applies connected components, filters tiny objects with `min_area`, and writes one sample folder per valid building. Each sample contains:

```text
pre.png
post.png
diff.png
mask.png
metadata.json
```

The metadata stores object-level information:

```text
building_id
sample_id
damage_class
area
perimeter
centroid
bbox
crop_bbox
polygon
disaster_type
```

The output dataset is organized by split and class:

```text
data/week11_buildings/
    train/no_damage/
    train/minor_damage/
    train/major_damage/
    train/destroyed/
    val/
    test/
```

Recommended extraction command:

```powershell
python src\week11\week11_extract_buildings.py --output-root data\week11_buildings --crop-size 96 --padding 12 --min-area 32
```

### Building-Level Dataset and Baseline

The dataset loader is:

```text
src/week11/week11_dataset.py
```

It returns normalized `pre`, `post`, and `diff` tensors plus the four-class damage label:

```text
no_damage
minor_damage
major_damage
destroyed
```

The first classifier is intentionally simple:

```text
src/week11/week11_model.py
```

It uses a shared ResNet18 encoder for the three crop branches. The fusion vector is:

```text
concat(features_pre, features_post, features_diff, abs(features_pre - features_post))
```

The head is a small MLP with ReLU and dropout. The loss is plain `CrossEntropyLoss`, because the object-level classification problem should be less imbalanced than dense pixel segmentation.

Training is handled by:

```text
src/week11/week11_train_classifier.py
```

Recommended first run:

```powershell
python src\week11\week11_train_classifier.py --dataset-root data\week11_buildings --epochs 20 --batch-size 32
```

Metrics include overall accuracy, macro F1, weighted F1, and per-class precision, recall, and F1. The most important early indicators are:

- `recall_minor_damage`
- `recall_major_damage`
- macro F1
- confusion between minor and major damage

### Week 11 Results Interpretation

The first object-level experiments show that the transition from dense segmentation to building-level classification is scientifically useful, but the minority damage classes remain difficult.

The strongest capped and augmented Week 11 run used:

```text
Siamese ResNet18 classifier
96x96 building crops
pre/post/difference inputs
effective class weighting
weighted sampler
training augmentation
no_damage capped at 5000 samples
all minor_damage, major_damage, and destroyed samples retained
```

The best validation checkpoint was epoch 23:

| Metric | Value |
|---|---:|
| accuracy | 0.9431 |
| macro F1 | 0.5067 |
| weighted F1 | 0.9524 |
| no_damage F1 | 0.9703 |
| minor_damage F1 | 0.0262 |
| major_damage F1 | 0.1561 |
| destroyed F1 | 0.8743 |
| minor_damage recall | 0.0441 |
| major_damage recall | 0.4286 |
| destroyed recall | 0.8922 |

The object-level classifier substantially improves no-damage and destroyed classification and begins to recover major-damage buildings under capped, augmented training. However, minor damage remains poorly separable, suggesting that the main bottleneck is not architecture but class scarcity and visual ambiguity.

This result is important because it changes the research diagnosis. The model is capable of learning strong object-level representations for clear classes, especially no damage and destroyed buildings. It also begins to identify major damage once the training distribution is controlled. Minor damage, however, has very few examples and is visually close to both no damage and major damage. Therefore, the next priority should be error analysis and data strategy rather than adding architectural complexity.

An additional inverse-weighted experiment was tested with a stronger minority-class correction:

```text
class_weight_mode = inverse
weighted sampler = enabled
training augmentation = enabled
no_damage capped at 3000 samples
```

This experiment increased minority recall, but it over-corrected the imbalance problem. At the best checkpoint, minor-damage recall rose to 0.4412 and major-damage recall rose to 0.6327, but precision collapsed to 0.0115 for minor damage and 0.0120 for major damage. The model predicted 2606 minor-damage buildings and 2574 major-damage buildings even though the validation set contained only 68 minor-damage and 49 major-damage buildings.

This confirms that aggressive inverse weighting does not solve the minority damage problem. It makes the classifier sensitive to minority classes, but not discriminative. The effective-weighted capped and augmented model remains the best Week 11 baseline because it preserves strong no-damage and destroyed performance while beginning to recover major-damage buildings.

### Error Analysis

Week 11 adds object-level qualitative inspection:

```text
src/week11/week11_error_analysis.py
```

This script loads the best classifier checkpoint, runs validation inference, writes a prediction record CSV, and saves pre/post/diff panels for misclassified building crops. This is a cleaner error-analysis target than pixel-level masks because each failure corresponds to one damaged building.

Recommended command:

```powershell
python src\week11\week11_error_analysis.py --dataset-root data\week11_buildings --checkpoint results\week11\checkpoints\week11_siamese_resnet18_best.pt
```

### Research Transition

Week 11 provides the bridge from segmentation to object-level reasoning:

- Segmentation remains the building localization backbone.
- Ground-truth polygons provide the first reliable object crops.
- Predicted masks can later replace ground-truth masks to test end-to-end deployment.
- Building-level confusion matrices make minor/major/destroyed ambiguity easier to study.
- The metadata prepares the project for later morphology, graph, and TDA features.

### Phase 6: Morphology and TDA Feature Fusion

Phase 6 introduces feature enrichment after establishing the object-level baseline. The implementation is intentionally conservative: it begins with classical morphology, change, and lightweight topological features before requiring external persistent-homology libraries.

The feature extractor is:

```text
src/week11/week11_features.py
```

It computes features from each building crop and mask, including:

- building area ratio
- metadata area and perimeter
- compactness
- bounding-box aspect ratio
- extent and solidity
- connected-component count
- hole count
- Euler number
- contour fragmentation
- distance-transform statistics
- pre/post edge-density change
- mean and high-percentile image difference
- high-difference connected components
- high-difference Euler number and contour fragmentation

These features provide the first topology-aware representation without adding a Gudhi or Ripser dependency. The topological part is not full persistent homology yet, but it captures related structural signals: connected regions, holes, fragmentation, and topology changes in high-difference regions.

The extractor also supports optional persistent-homology features through Gudhi. If Gudhi is installed, cubical-complex persistence summaries are added for the building mask and high-difference regions:

- dimension-0 persistence count
- dimension-0 persistence entropy
- dimension-0 mean lifetime
- dimension-1 persistence count
- dimension-1 persistence entropy

If Gudhi is not installed, these persistent-homology fields are filled with zeros so the training pipeline remains runnable and the feature vector dimension stays fixed.

The fusion model is:

```text
SiameseBuildingFeatureClassifier
```

It keeps the same ResNet18 Siamese image encoder and concatenates a learned handcrafted-feature embedding with the CNN embedding:

```text
final_feature = concat(
    cnn_pre,
    cnn_post,
    cnn_diff,
    abs(cnn_pre - cnn_post),
    morphology_tda_embedding
)
```

Recommended Phase 6 run:

```powershell
python src\week11\week11_train_classifier.py --dataset-root data\week11_buildings --results-dir results\week11_feature_fusion --epochs 25 --batch-size 32 --class-weight-mode effective --weighted-sampler --augment-train --max-train-per-class 5000 -1 -1 3091 --use-handcrafted-features
```

The Phase 6 run included morphology, change, topology-inspired features, and Gudhi cubical-complex persistent-homology summaries. The best validation checkpoint was epoch 7:

| Metric | CNN capped augmented baseline | Morphology + TDA feature fusion |
|---|---:|---:|
| accuracy | 0.9431 | 0.5507 |
| macro F1 | 0.5067 | 0.4033 |
| weighted F1 | 0.9524 | 0.6878 |
| no_damage F1 | 0.9703 | 0.6796 |
| minor_damage precision | 0.0186 | 0.0078 |
| minor_damage recall | 0.0441 | 0.5588 |
| minor_damage F1 | 0.0262 | 0.0154 |
| major_damage precision | 0.0955 | 0.0540 |
| major_damage recall | 0.4286 | 0.5102 |
| major_damage F1 | 0.1561 | 0.0977 |
| destroyed F1 | 0.8743 | 0.8207 |

The morphology and persistent-homology feature-fusion experiment did not outperform the Siamese CNN baseline. It increased minority-class recall, especially for minor damage, but it also caused severe minority overprediction. The validation set contained only 68 minor-damage buildings, but the feature-fusion model predicted 4880 buildings as minor damage. This produced a minor-damage precision of only 0.0078. The same pattern appeared for major damage: recall improved slightly, but precision remained low because the model predicted 463 major-damage buildings for only 49 true examples.

This means the feature-fusion model learned a broad structural-change signal, but it did not learn reliable damage-severity discrimination. Persistent-homology and morphology features made the classifier more sensitive to possible damage, but not more precise about whether a building was no-damage, minor-damage, major-damage, or destroyed.

### Final Week 11 Conclusion

The final Week 11 baseline is the capped and augmented Siamese ResNet18 classifier:

```text
results/week11_capped_augmented
```

This model is the strongest overall object-level system because it preserves high no-damage and destroyed performance while beginning to recover major-damage buildings. Aggressive inverse weighting and morphology + TDA feature fusion both increased minority recall, but they did so by overpredicting minority classes and reducing precision, accuracy, and weighted F1.

The main Week 11 conclusion is that object-level classification is a better research direction than dense pixel-level damage segmentation for this project, but the key remaining bottleneck is minor-damage separability. Minor damage is rare, visually subtle, and close to both no-damage and major-damage cases. The next research work should therefore focus on data quality, label ambiguity, calibrated feature fusion, and targeted error analysis rather than simply adding stronger class weights or more complex features.

## Week 12: Advanced Object-Level Damage Representation Learning

Week 12 changes the project goal from segmentation optimization to embedding-centric object-level damage understanding. The Week 11 object crop dataset remains the input, but the scientific target is now the geometry of the learned building representation.

The core Week 12 question is:

```text
How can the model improve semantic separability between visually ambiguous
building-level damage classes?
```

The most important ambiguity is:

```text
minor_damage vs major_damage
```

Minor damage is also frequently confused with no damage because it can appear as subtle roof texture change, small debris, or weak structural cues. This means the remaining failure is not mainly a localization problem. It is an embedding-overlap, class-definition, and data-distribution problem.

### Week 12 Phase 1: Stronger CNN Backbones

The Week 11 classifier uses a shared ResNet18 encoder. Week 12 introduces stronger encoders while preserving the same object-level training protocol:

```text
src/week12/week12_model_backbones.py
src/week12/week12_train_backbone.py
src/week12/week12_eval_backbone.py
```

Supported backbones:

```text
resnet18
resnet34
efficientnet_b0
convnext_tiny
```

The recommended experimental order is:

```text
12A-1: ResNet34
12A-2: EfficientNet-B0
12A-3: ConvNeXt-Tiny
```

ResNet34 is the safest upgrade because it adds depth and texture capacity with minimal disruption. EfficientNet-B0 is expected to help subtle local roof patterns, which may improve minor-damage precision. ConvNeXt-Tiny is the strongest long-term representation candidate because it provides modern spatial features, but it is slower and may need more tuning.

Example ResNet34 run:

```text
python src/week12/week12_train_backbone.py --dataset-root data/week11_buildings --results-dir results/week12_resnet34 --backbone resnet34 --epochs 25 --batch-size 32 --class-weight-mode effective --weighted-sampler --augment-train --max-train-per-class 5000 -1 -1 3091
```

### Week 12 Phase 2: Embedding and Metric Learning

Cross-entropy learns a decision boundary, but it does not explicitly enforce compact within-class clusters or large between-class distances. This is a poor match for the Week 11 failure mode, where minor-damage embeddings overlap with no-damage and major-damage examples.

Week 12 therefore adds metric-learning objectives:

```text
ArcFace
Supervised contrastive learning
```

ArcFace is the first recommended metric-learning experiment because it adds an angular margin between classes. This should encourage stronger separation among minor, major, and destroyed buildings:

```text
python src/week12/week12_train_backbone.py --dataset-root data/week11_buildings --results-dir results/week12_arcface_resnet34 --backbone resnet34 --loss-type arcface --fusion concat --epochs 25 --batch-size 32 --weighted-sampler --augment-train --max-train-per-class 5000 -1 -1 3091
```

Supervised contrastive learning is implemented as an embedding pretraining option:

```text
python src/week12/week12_train_backbone.py --dataset-root data/week11_buildings --results-dir results/week12_supcon_resnet34 --backbone resnet34 --loss-type supcon --epochs 25 --batch-size 32 --weighted-sampler --augment-train --max-train-per-class 5000 -1 -1 3091
```

Embedding visualization is a required Week 12 analysis artifact. The evaluator exports raw embeddings and projection plots:

```text
python src/week12/week12_eval_backbone.py --dataset-root data/week11_buildings --checkpoint results/week12_arcface_resnet34/checkpoints/week12_resnet34_concat_arcface_best.pt --output-dir results/week12_arcface_resnet34/eval
```

The evaluator always saves PCA plots. If optional packages are installed, it also saves t-SNE and UMAP plots. These plots should show whether minor and major damage form separable clusters or remain mixed.

### Week 12 Phase 3: Hierarchical Damage Classification

The xBD labels are naturally hierarchical:

```text
Stage 1: no_damage vs damaged
Stage 2: minor_damage vs major_damage vs destroyed
```

Week 12 implements this decomposition in:

```text
src/week12/week12_hierarchical_model.py
src/week12/week12_train_stage1.py
src/week12/week12_train_stage2.py
src/week12/week12_inference_pipeline.py
```

The motivation is that the direct four-way classifier forces minor damage to compete against the extremely dominant no-damage class. The hierarchical design removes that interference in Stage 2 and lets the damaged classifier focus only on damage severity.

Stage 1 should prioritize high damaged recall:

```text
python src/week12/week12_train_stage1.py --dataset-root data/week11_buildings --results-dir results/week12_hierarchical_stage1 --backbone resnet34 --fusion gated --epochs 20 --batch-size 32 --weighted-sampler --augment-train --damaged-weight 2.0
```

Stage 2 trains only on damaged buildings:

```text
python src/week12/week12_train_stage2.py --dataset-root data/week11_buildings --results-dir results/week12_hierarchical_stage2 --backbone resnet34 --fusion gated --epochs 25 --batch-size 32 --weighted-sampler --augment-train
```

The full inference pipeline combines both stages with a tunable damaged threshold:

```text
python src/week12/week12_inference_pipeline.py --dataset-root data/week11_buildings --stage1-checkpoint results/week12_hierarchical_stage1/checkpoints/week12_stage1_best.pt --stage2-checkpoint results/week12_hierarchical_stage2/checkpoints/week12_stage2_best.pt --damaged-threshold 0.35 --output-dir results/week12_hierarchical_pipeline
```

Lowering the damaged threshold prioritizes recall, which is appropriate because missing a damaged building is more costly than sending a borderline building to Stage 2.

### Week 12 Phase 4: Temporal Attention Fusion

Week 11 uses shallow feature concatenation:

```text
concat(pre, post, diff, abs(pre - post))
```

Week 12 adds two learned fusion alternatives:

```text
gated
cross_attention
```

Gated fusion learns how much the post-disaster embedding should replace the pre-disaster embedding. Cross-attention allows the pre, post, difference, and absolute-change embeddings to interact before classification. This is more aligned with the object-level problem because the model can learn where and how change matters rather than relying only on static concatenation.

### Week 12 Expected Evaluation

Week 12 should be compared against the strongest Week 11 baseline:

```text
results/week11_capped_augmented
```

The most important metrics are:

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

The expected scientific outcome is not only a higher score, but a clearer explanation of whether stronger representations reduce the overlap between minor and major damage embeddings.

### Week 12 Results

The final Week 12 experiments used the expanded object-level dataset:

```text
data/week11_buildings_week8_extra
```

This means Week 12 results should be interpreted as object-level representation learning on the Week 11 crop pipeline with the additional Week 8 minority-class expansion data. The strongest Week 11 reference point remains:

```text
results/week11_capped_augmented
```

with:

```text
accuracy:              0.9431
macro F1:              0.5067
weighted F1:           0.9524
minor_damage F1:       0.0262
major_damage F1:       0.1561
destroyed F1:          0.8743
```

#### ResNet34 Baseline

The first Week 12 backbone test used ResNet34 with effective class weighting and no weighted sampler:

```text
python src/week12/week12_train_backbone.py --dataset-root data/week11_buildings_week8_extra --results-dir results/week12/week12_resnet34_effective_no_sampler --backbone resnet34 --epochs 25 --batch-size 32 --class-weight-mode effective --augment-train --max-train-per-class 5000 -1 -1 3091
```

Best checkpoint:

```text
epoch 25
```

Metrics:

```text
accuracy:              0.9698
macro F1:              0.5699
weighted F1:           0.9692

no_damage F1:          0.9844
minor_damage F1:       0.0536
major_damage F1:       0.3333
destroyed F1:          0.9083

minor precision:       0.0682
minor recall:          0.0441
major precision:       0.3208
major recall:          0.3469
predicted_minor:       44
predicted_major:       53
```

This was a positive result. ResNet34 improved macro F1, major-damage F1, destroyed F1, and overall accuracy compared with the Week 11 ResNet18 object-level baseline. The important scientific point is that the model did not simply overpredict minority classes: the predicted minor and major counts remained close to their validation supports.

An earlier ResNet34 run using both effective class weighting and a weighted sampler collapsed into severe `major_damage` overprediction:

```text
accuracy:              0.0199
macro F1:              0.0752
predicted_no_damage:   0
predicted_major:       12119
```

This showed that weighted sampling plus class-weighted loss can over-correct the imbalance problem for stronger Week 12 backbones. The remaining backbone experiments therefore removed the weighted sampler.

#### EfficientNet-B0 Baseline

EfficientNet-B0 was tested because it is often strong on local texture and small visual patterns. However, the result was negative:

```text
best_epoch:            23
accuracy:              0.5948
macro F1:              0.3682
weighted F1:           0.7053

minor_damage F1:       0.0162
major_damage F1:       0.0387
destroyed F1:          0.7055

minor recall:          0.3971
minor precision:       0.0083
predicted_minor:       3271
support_minor:         68

major recall:          0.3469
major precision:       0.0205
predicted_major:       829
support_major:         49
```

EfficientNet-B0 became highly sensitive to possible damage but not precise. It increased minority recall by flooding the validation set with minority predictions. This is not improved semantic separability; it is minority overprediction.

#### ConvNeXt-Tiny Baseline

ConvNeXt-Tiny was the strongest backbone experiment:

```text
accuracy:              0.9617
macro F1:              0.5803
weighted F1:           0.9623

no_damage F1:          0.9800
minor_damage F1:       0.0787
major_damage F1:       0.3878
destroyed F1:          0.8747

minor precision:       0.0847
minor recall:          0.0735
predicted_minor:       59
support_minor:         68

major precision:       0.3878
major recall:          0.3878
predicted_major:       49
support_major:         49
```

ConvNeXt-Tiny achieved the best macro F1 and the strongest combined minor/major behavior. The predicted minor and major counts were very close to the true validation supports, which is important because it indicates improved separability rather than a recall-only overprediction effect.

Compared with ResNet34, ConvNeXt-Tiny improved:

```text
macro F1:              0.5699 -> 0.5803
minor_damage F1:       0.0536 -> 0.0787
major_damage F1:       0.3333 -> 0.3878
```

The main tradeoff was destroyed-class performance:

```text
destroyed F1:          0.9083 -> 0.8747
accuracy:              0.9698 -> 0.9617
```

For the Week 12 research question, ConvNeXt-Tiny is still the better result because the goal is to improve semantic separability for ambiguous damage classes, especially `minor_damage` and `major_damage`.

#### ArcFace Metric Learning

ArcFace was tested using ConvNeXt-Tiny as the backbone:

```text
best_epoch:            18
accuracy:              0.9485
macro F1:              0.4948
weighted F1:           0.9524

minor_damage F1:       0.0750
major_damage F1:       0.0833
destroyed F1:          0.8477

minor recall:          0.1324
minor precision:       0.0523
major recall:          0.0612
major precision:       0.1304
```

ArcFace did not improve the Week 12 embedding space. It increased minor-damage recall slightly, but it severely reduced major-damage recognition:

```text
major F1:              0.3878 -> 0.0833
major recall:          0.3878 -> 0.0612
```

This is a negative result. Plain ConvNeXt-Tiny with cross-entropy remains the best Week 12 end-to-end model.

### Week 12 Hierarchical Classification Results

The hierarchical experiments used ConvNeXt-Tiny with gated temporal fusion. The goal was to test whether removing `no_damage` from the damaged-class decision space improves minor/major/destroyed separation.

#### Stage 1: No Damage vs Damaged

Stage 1 was trained as a binary classifier:

```text
no_damage vs damaged
```

Best checkpoint:

```text
epoch 5
```

Metrics:

```text
accuracy:              0.9297
macro F1:              0.8520

no_damage precision:   0.9924
no_damage recall:      0.9283
no_damage F1:          0.9593

damaged precision:     0.6159
damaged recall:        0.9419
damaged F1:            0.7448

support_damaged:       1360
predicted_damaged:     2080
```

This is a successful Stage 1 result because the priority is high damaged recall. The model catches most damaged buildings and sends extra borderline no-damage buildings to Stage 2. That is acceptable for a recall-oriented first stage.

#### Stage 2: Minor vs Major vs Destroyed

Stage 2 was trained only on damaged buildings:

```text
minor_damage vs major_damage vs destroyed
```

Best checkpoint:

```text
epoch 20
```

Metrics:

```text
accuracy:              0.9529
macro F1:              0.7427

minor_damage precision:0.7800
minor_damage recall:   0.5735
minor_damage F1:       0.6610
predicted_minor:       50
support_minor:         68

major_damage precision:0.6279
major_damage recall:   0.5510
major_damage F1:       0.5870
predicted_major:       43
support_major:         49

destroyed precision:   0.9708
destroyed recall:      0.9895
destroyed F1:          0.9801
```

This is one of the most important Week 12 findings. When `no_damage` is removed from the decision space, the damaged classes become much more separable. Stage 2 does not show minority flooding: predicted minor and major counts remain close to support. This strongly supports the hypothesis that much of the four-way classifier difficulty comes from no-damage interference and hierarchy/calibration, not from a complete absence of visual damage cues.

#### Full Hierarchical Pipeline Threshold Sweep

The full hierarchy combines Stage 1 and Stage 2. A damaged threshold controls how likely Stage 1 must be to send a building into Stage 2.

Threshold `0.35`:

```text
accuracy:              0.9025
macro F1:              0.4935
weighted F1:           0.9180

minor_damage F1:       0.1070
major_damage F1:       0.1695
destroyed F1:          0.7529
predicted_minor:       175
predicted_major:       246
predicted_destroyed:   1998
```

Threshold `0.50`:

```text
accuracy:              0.9265
macro F1:              0.5181
weighted F1:           0.9355

minor_damage F1:       0.1136
major_damage F1:       0.2025
destroyed F1:          0.7968
predicted_minor:       108
predicted_major:       188
predicted_destroyed:   1784
```

Threshold `0.65`:

```text
accuracy:              0.9442
macro F1:              0.5293
weighted F1:           0.9485

minor_damage F1:       0.0451
major_damage F1:       0.2682
destroyed F1:          0.8341
predicted_minor:       65
predicted_major:       130
predicted_destroyed:   1627
```

Threshold `0.75`:

```text
accuracy:              0.9558
macro F1:              0.5567
weighted F1:           0.9570

minor_damage F1:       0.0583
major_damage F1:       0.3358
destroyed F1:          0.8563
predicted_minor:       35
predicted_major:       88
predicted_destroyed:   1527
```

The best full hierarchical pipeline was threshold `0.75`, but it remained below the plain ConvNeXt-Tiny end-to-end classifier:

```text
plain ConvNeXt-Tiny macro F1:   0.5803
hierarchy threshold 0.75:       0.5567
```

The threshold sweep reveals the central hierarchy tradeoff:

```text
low threshold  -> high damaged recall but too many false damaged predictions
high threshold -> better precision but subtle minor damage is filtered out
```

Thus, the hierarchy is scientifically valuable but not the best deployed end-to-end Week 12 model. Its strongest contribution is diagnostic: Stage 2 proves that damaged-class separability improves sharply once the overwhelming no-damage class is removed.

### Week 12 Phase 4 Results: Temporal Attention Fusion

The final Week 12 experiments tested whether learned temporal fusion can improve over the static ConvNeXt-Tiny concatenation baseline.

The baseline fusion was:

```text
concat(pre, post, diff, abs(pre - post))
```

The two learned alternatives were:

```text
gated
cross_attention
```

#### Gated Fusion

Gated fusion learns how much the post-disaster embedding should replace or modify the pre-disaster embedding.

Metrics:

```text
best_epoch:            11
accuracy:              0.9580
macro F1:              0.5759
weighted F1:           0.9596

no_damage F1:          0.9777
minor_damage F1:       0.1250
major_damage F1:       0.3333
destroyed F1:          0.8675

minor precision:       0.1333
minor recall:          0.1176
predicted_minor:       60
support_minor:         68

major precision:       0.2727
major recall:          0.4286
predicted_major:       77
support_major:         49
```

Gated fusion produced the strongest `minor_damage` result of Week 12:

```text
concat minor F1:       0.0787
gated minor F1:        0.1250
```

This is important because minor damage is the project bottleneck. The macro-F1 tradeoff compared with concat was very small:

```text
concat macro F1:       0.5803
gated macro F1:        0.5759
difference:            0.0044
```

Therefore, gated fusion is the most scientifically useful Week 12 model for improving subtle damage recognition, even though it is not the top model by macro F1.

#### Cross-Attention Fusion

Cross-attention lets the pre, post, difference, and absolute-change embeddings interact before classification.

Metrics:

```text
best_epoch:            25
accuracy:              0.9598
macro F1:              0.5793
weighted F1:           0.9597

no_damage F1:          0.9789
minor_damage F1:       0.0583
major_damage F1:       0.4222
destroyed F1:          0.8580

minor precision:       0.0857
minor recall:          0.0441
predicted_minor:       35
support_minor:         68

major precision:       0.4634
major recall:          0.3878
predicted_major:       41
support_major:         49
```

Cross-attention produced the strongest `major_damage` result:

```text
concat major F1:       0.3878
cross-attention F1:    0.4222
```

However, it reduced minor-damage F1:

```text
concat minor F1:       0.0787
cross-attention F1:    0.0583
```

This suggests that cross-attention helps more visible structural damage patterns but does not help the subtle minor-damage class.

#### Phase 4 Interpretation

The Phase 4 comparison is:

```text
ConvNeXt-Tiny + concat:
macro F1:              0.5803
minor_damage F1:       0.0787
major_damage F1:       0.3878
destroyed F1:          0.8747

ConvNeXt-Tiny + gated:
macro F1:              0.5759
minor_damage F1:       0.1250
major_damage F1:       0.3333
destroyed F1:          0.8675

ConvNeXt-Tiny + cross-attention:
macro F1:              0.5793
minor_damage F1:       0.0583
major_damage F1:       0.4222
destroyed F1:          0.8580
```

Static concat remains the best model by macro F1. However, gated fusion is selected as the preferred Week 12 model because it substantially improves the main bottleneck class, `minor_damage`, with only a negligible macro-F1 decrease. Cross-attention is a useful secondary result because it improves `major_damage`, but it does not address the minor-damage bottleneck.

### Final Week 12 Ranking

The Week 12 experiments can be ranked in two ways.

By macro F1:

```text
1. ConvNeXt-Tiny CE, effective weighting, no sampler
   macro F1: 0.5803

2. ConvNeXt-Tiny cross-attention CE
   macro F1: 0.5793

3. ConvNeXt-Tiny gated CE
   macro F1: 0.5759

4. ResNet34 CE, effective weighting, no sampler
   macro F1: 0.5699

5. Hierarchical ConvNeXt, threshold 0.75
   macro F1: 0.5567

6. ConvNeXt-Tiny ArcFace
   macro F1: 0.4948

7. EfficientNet-B0 CE
   macro F1: 0.3682

8. ResNet34 with effective weighting and weighted sampler
   macro F1: 0.0752
```

By the project's main scientific bottleneck, minor-damage separability:

```text
1. ConvNeXt-Tiny gated CE
   minor_damage F1: 0.1250

2. ConvNeXt-Tiny concat CE
   minor_damage F1: 0.0787

3. ConvNeXt-Tiny ArcFace
   minor_damage F1: 0.0750

4. ConvNeXt-Tiny cross-attention CE
   minor_damage F1: 0.0583
```

The best Week 12 model by macro F1 is:

```text
results/week12/week12_convnext_tiny_effective_no_sampler
```

with:

```text
accuracy:              0.9617
macro F1:              0.5803
weighted F1:           0.9623
minor_damage F1:       0.0787
major_damage F1:       0.3878
destroyed F1:          0.8747
```

The preferred Week 12 model for the project goal is:

```text
results/week12/week12_convnext_tiny_gated_effective_no_sampler
```

with:

```text
accuracy:              0.9580
macro F1:              0.5759
weighted F1:           0.9596
minor_damage F1:       0.1250
major_damage F1:       0.3333
destroyed F1:          0.8675
```

### Final Week 12 Conclusion

Week 12 confirms that the project has moved from dense segmentation optimization to semantic building-level representation learning. The strongest backbone improvement came from ConvNeXt-Tiny, which improved minor and major damage classification without causing minority-class flooding. ResNet34 was also a strong positive result, while EfficientNet-B0 and ArcFace were negative results under the tested settings.

The hierarchical experiment provides the clearest scientific insight. Stage 2 achieved strong damaged-class performance when `no_damage` was removed, proving that object crops contain useful severity cues for minor, major, and destroyed buildings. However, the full hierarchy remained limited by Stage 1 threshold calibration: permissive thresholds over-send no-damage buildings into Stage 2, while strict thresholds suppress subtle minor-damage examples.

The temporal-fusion experiment shows that learned fusion changes class-specific behavior. Cross-attention improves major-damage F1, while gated fusion gives the best minor-damage F1 of all Week 12 end-to-end models. Because minor damage is the central bottleneck, gated ConvNeXt-Tiny is selected as the preferred Week 12 model, even though static concat has the highest macro F1 by a very small margin.

The final Week 12 conclusion is:

```text
ConvNeXt-Tiny with concat fusion is the best Week 12 model by macro F1, but
ConvNeXt-Tiny with gated temporal fusion is the preferred Week 12 model for the
project's scientific goal because it gives the strongest minor-damage F1. The
hierarchical Stage 2 result remains the strongest diagnostic evidence that the
remaining bottleneck is not localization or model capacity alone. The core
challenge is calibrated semantic separation under class imbalance and label
ambiguity, especially for minor damage.
```

## Week 13: Topology-Guided Semantic Calibration

Week 13 changes direction from training another classifier to adding a narrow verifier for the project’s clearest semantic ambiguity:

```text
no_damage <-> minor_damage
```

This is intentional. Earlier experiments showed that handcrafted topology and morphology are not strong enough to solve the full four-way xBD damage task. Week 13 therefore does not ask TDA to classify every damage class. Instead, it uses topology only where it is scientifically plausible to help: ambiguous CNN decisions between visually intact buildings and subtly damaged buildings.

The Week 13 objective is:

```text
Use topology as a post-hoc verifier for no_damage/minor_damage CNN mistakes,
then measure whether it improves minor_damage precision, recall, and F1.
```

The baseline remains:

```text
ConvNeXt-Tiny + gated fusion
```

The Week 13 hybrid model is:

```text
ConvNeXt-Tiny + gated fusion -> TDA verifier -> corrected prediction
```

### Week 13 Phase 1: Error Region Isolation

The first phase isolates the only error region that Week 13 is allowed to correct:

```text
true no_damage predicted minor_damage
true minor_damage predicted no_damage
```

This is implemented in:

```text
src/week13/week13_error_isolation.py
```

Example:

```powershell
python src\week13\week13_error_isolation.py --dataset-root data\week11_buildings_week8_extra --checkpoint results\week12\week12_convnext_tiny_gated_effective_no_sampler\checkpoints\week12_convnext_tiny_gated_ce_best.pt --split val --output-csv results\week13_topology\error_regions.csv
```

The output CSV records:

```text
metadata_path
sample_dir
true label
CNN prediction
p_no_damage
p_minor_damage
whether the row is a target-pair error
```

This phase keeps the experiment honest: Week 13 is not a broad TDA classifier. It is a correction module for one known semantic failure mode.

### Week 13 Phase 2: Persistent-Homology-Inspired Topology Pipeline

Topology signatures are extracted in:

```text
src/week13/week13_topology_features.py
```

For each object crop, Week 13 builds topology summaries from:

```text
building mask
edge map
difference mask
```

If a saved building mask exists in the crop directory, it is used. Otherwise, the script estimates a building mask from the post-disaster crop using Otsu thresholding. Edge maps are generated with Canny edges, and difference masks are generated from the saved `diff.png` crop.

The topology pipeline exports:

```text
Betti-0 curves
Betti-1 curves
persistence-style diagrams from Betti curve changes
Wasserstein distances between edge and difference diagrams
bottleneck distances between edge and difference diagrams
area and ratio summaries
```

The implementation is intentionally lightweight and dependency-minimal. It uses OpenCV connected components and hole counting as a practical image-filtration proxy for persistent homology, avoiding a hard dependency on specialized TDA libraries.

Generate topology features with:

```powershell
python src\week13\week13_fit_topology_threshold.py --dataset-root data\week11_buildings_week8_extra --split val --topology-csv results\week13_topology\topology_features.csv --output-dir results\week13_topology\threshold
```

### Week 13 Phase 3: Topology Threshold Calibration

The calibration phase learns a single scalar decision rule:

```text
topology_distance_threshold
```

The score is:

```text
distance_to_no_damage_topology_prototype - distance_to_minor_damage_topology_prototype
```

Interpretation:

```text
higher score -> topology is closer to minor_damage
lower score  -> topology is closer to no_damage
```

The threshold is selected to best separate true `no_damage` from true `minor_damage` on the calibration split. The fitted configuration is saved as:

```text
results/week13_topology/threshold/topology_threshold.json
```

This file contains:

```text
feature names
normalization statistics
no_damage topology prototype
minor_damage topology prototype
topology_distance_threshold
precision / recall / F1 for no-vs-minor topology separation
```

### Week 13 Phase 4: Hybrid CNN + TDA Correction

The final Week 13 pipeline applies a constrained correction rule:

```text
1. Run the ConvNeXt-Tiny gated CNN.
2. If the CNN prediction is no_damage or minor_damage, check whether the prediction is ambiguous.
3. Extract topology features for that crop.
4. Apply the topology_distance_threshold.
5. Correct only between no_damage and minor_damage.
6. Leave major_damage and destroyed untouched.
```

This is implemented in:

```text
src/week13/week13_hybrid_correction.py
```

Example:

```powershell
python src\week13\week13_hybrid_correction.py --dataset-root data\week11_buildings_week8_extra --checkpoint results\week12\week12_convnext_tiny_gated_effective_no_sampler\checkpoints\week12_convnext_tiny_gated_ce_best.pt --threshold-json results\week13_topology\threshold\topology_threshold.json --split val --output-dir results\week13_topology\hybrid --ambiguity-margin 0.20
```

The `--ambiguity-margin` controls how close `p_no_damage` and `p_minor_damage` must be before TDA is allowed to intervene. This prevents topology from overriding confident CNN predictions.

### Week 13 Phase 5: Scientific Evaluation

Week 13 compares:

```text
Baseline:
ConvNeXt-Tiny gated

Hybrid:
ConvNeXt-Tiny gated + TDA correction
```

The primary metrics are:

```text
minor_damage precision
minor_damage recall
minor_damage F1
predicted_minor_damage count
no_damage false positives
minor_damage false negatives
```

The hybrid script saves:

```text
metrics/hybrid_metrics.json
confusion_matrices/baseline_confusion_matrix.csv
confusion_matrices/hybrid_confusion_matrix.csv
confusion_matrices/hybrid_confusion_matrix.png
corrections.csv
```

The most important scientific success condition is not just a higher minor recall. A good Week 13 result should improve `minor_damage` F1 without flooding the validation set with minor predictions.

### Week 13 Final Results and Interpretation

The Week 13 results are scientifically valuable even though the final hybrid system did not improve the main benchmark metric. This is a strong negative result rather than a failed experiment: the proposed TDA refinement mechanism produced interpretable topological corrections, but it did not outperform the learned deep representation baseline on the final benchmark.

The baseline gated ConvNeXt-Tiny model achieved:

```text
accuracy:              0.9580
macro F1:              0.5759
weighted F1:           0.9596
minor_damage F1:       0.1250
minor_damage precision:0.1333
minor_damage recall:   0.1176
predicted_minor:       60
```

The hybrid CNN + TDA correction achieved:

```text
accuracy:              0.9593
macro F1:              0.5595
weighted F1:           0.9599
minor_damage F1:       0.0588
minor_damage precision:0.0882
minor_damage recall:   0.0441
predicted_minor:       34
num_corrections:       28
```

The hybrid system slightly increased overall accuracy and weighted F1, but these metrics are dominated by the large `no_damage` class. The most important result is that macro F1 decreased and `minor_damage` F1 dropped from `0.1250` to `0.0588`. Therefore, the topology correction stage hurt minor-damage recognition under the tested rule.

This does not mean the TDA idea was invalid. Instead, it reveals a deeper scientific conclusion:

```text
The CNN learned stronger semantic representations for subtle damage than the
handcrafted topology verifier.
```

The topology verifier behaved mainly as a conservative filter. It reduced the number of predicted `minor_damage` buildings:

```text
predicted_minor_damage: 60 -> 34
minor_damage recall:   0.1176 -> 0.0441
```

This means the TDA module mostly suppressed minor predictions instead of recovering hidden minor-damage cases. The correction rule improved no-damage conservatism, but it removed too many true minor-damage examples.

The threshold calibration confirms the same pattern:

```text
topology threshold F1:        0.1111
topology threshold precision: 0.0893
topology threshold recall:    0.1471
```

These values show that topology-space separation between `no_damage` and `minor_damage` is weak. Persistent-homology-inspired features captured some meaningful structural change, but the distributions overlapped too much for a direct threshold rule to be reliable.

There is still important positive evidence. The topology prototypes are not random. For example, `edge_components_mean` differs strongly between the two learned prototypes:

```text
no_damage prototype:     -0.005
minor_damage prototype:   0.863
```

This indicates that TDA is detecting real geometric differences. However, only some minor-damage cases show visible topological deformation.

The core reason is that minor damage is often not topological. Many minor-damage examples in xBD are not primarily structural deformations. They may involve:

- discoloration
- small cracks
- tiny debris
- texture variation
- subtle roof changes

Persistent homology is better aligned with:

- connectivity
- holes
- connected components
- fragmentation
- structural deformation

This mismatch explains why the hybrid verifier struggled. It searched for structural topology change, while many `minor_damage` examples are appearance-level or texture-level changes without a strong change in topology.

The best Week 13 conclusion is:

```text
The topology-guided correction module produced interpretable geometric change
measurements but did not improve minor-damage recognition. The results suggest
that persistent homology captures structural deformation and fragmentation,
which are more aligned with major and destroyed damage than with subtle minor
damage. Therefore, the remaining minor-damage bottleneck appears to depend more
on semantic and texture-level temporal cues than on explicit topological change.
```

This creates a coherent research progression across the project:

```text
Week 10-11:
Dense segmentation struggles with ambiguous damage.

Week 11-12:
Object-level semantic representation learning performs better.

Week 12:
Gated temporal fusion improves subtle damage recognition.

Week 13:
Pure topological verification alone cannot reliably separate minor damage,
because minor damage is often semantic or texture-level rather than structural
topology-level.
```

Week 13 is therefore a meaningful negative result. It shows that topology is interpretable and can measure structural deformation, but it should not replace or directly override the learned CNN representation for subtle `minor_damage` classification. A better future use would be to integrate topology as an auxiliary feature, diagnostic signal, or regularizer rather than as a hard post-processing threshold.

### Week 13 Scientific Hypothesis

The Week 13 hypothesis is:

```text
CNN representations are best for broad semantic damage recognition,
but topology can act as a targeted verifier for subtle no_damage/minor_damage ambiguity.
```

This is a much narrower and stronger TDA claim than earlier handcrafted-feature experiments. Week 13 does not claim that topology solves xBD damage classification. It tests whether topology can improve the one region where semantic ambiguity is highest and where minor-damage performance matters most.

## Week 14: CrisisMMD v2 Social-Media Crisis Understanding

Week 14 extends the project beyond satellite imagery by adding a Twitter-based crisis understanding module using the local CrisisMMD v2 dataset:

```text
data/CrisisMMD_v2.0
```

This changes the system from a satellite-only damage classifier into a multi-source disaster intelligence pipeline. Satellite imagery estimates physical damage, topology features validate structural patterns, and social-media text adds real-time humanitarian context.

The Week 14 implementation is located in:

```text
src/week14/week14_prepare_crisismmd.py
src/week14/week14_emotion_pseudolabels.py
src/week14/week14_train_text_classifier.py
```

### Week 14 Task Mapping

The final CrisisMMD v2 mapping uses the corrected task definitions.

Task 1 is binary informativeness classification:

```text
input: tweet_text
label: text_info

classes:
- informative
- not_informative
```

The removed `dont_know_or_cant_judge` label is treated as an ignored label if it appears in raw files.

Task 2 is single-label humanitarian classification:

```text
input: tweet_text
label: text_human

classes:
- affected_individuals
- infrastructure_and_utility_damage
- injured_or_dead_people
- missing_or_found_people
- rescue_volunteering_or_donation_effort
- vehicle_damage
- other_relevant_information
- not_humanitarian
```

This is modeled as an eight-class multi-class classification problem. The `not_humanitarian` class is kept as a real v2 label instead of being discarded, because it makes the classifier useful on mixed crisis streams where many tweets are not actionable humanitarian reports.

Task 3 is damage-severity proxy classification:

```text
input: tweet_text
label: image_damage

classes:
- severe_damage
- mild_damage
- little_or_no_damage
```

Although the CrisisMMD column is named `image_damage`, Week 14 uses it as a text-only proxy target. This is a valid research setup because the tweet text and attached image describe the same crisis post. Null damage rows and ambiguous `dont_know_or_cant_judge` labels are removed from the training set.

### Week 14 Data Preparation

The preparation script reads all event TSV files from `data/CrisisMMD_v2.0/annotations`, normalizes labels, deduplicates text-level tasks by `tweet_id`, and writes task-specific CSV files:

```text
results/week14_crisismmd/processed/informativeness.csv
results/week14_crisismmd/processed/humanitarian.csv
results/week14_crisismmd/processed/damage_severity.csv
results/week14_crisismmd/processed/emotion_prompts.jsonl
results/week14_crisismmd/processed/label_maps.json
results/week14_crisismmd/processed/summary.json
```

The default split is event-based to reduce leakage:

```text
train: all events except validation and test events
val:   mexico_earthquake
test:  srilanka_floods
```

This avoids mixing the same disaster event across train and test under the default setup. The validation and test events can be changed with `--val-events` and `--test-events`.

The generated Week 14 task sizes are:

```text
informativeness:
train 13990, val 1238, test 830

humanitarian:
train 13990, val 1238, test 830

damage_severity:
train 2950, val 165, test 83
```

The humanitarian label distribution confirms the expected CrisisMMD imbalance:

```text
other_relevant_information:              5951
not_humanitarian:                        4556
rescue_volunteering_or_donation_effort:  3292
infrastructure_and_utility_damage:       1208
injured_or_dead_people:                   486
affected_individuals:                     471
vehicle_damage:                            54
missing_or_found_people:                   40
```

Because the minority classes are very small, Week 14 training supports class-weighted cross-entropy and focal loss.

### Week 14 Emotion Detection

Experiment C adds emotion detection through pseudo-labeling. The emotion set is:

```text
Fear
Sadness
Anger
Hope
Neutral
```

The preparation script writes `emotion_prompts.jsonl` with one prompt per tweet:

```text
Task: Classify the crisis tweet emotion into exactly one label from
[Fear, Sadness, Anger, Hope, Neutral]. Answer only one label.

Tweet: <tweet_text>
```

The pseudo-label filtering script accepts JSONL outputs from one or more LLM labeling passes and applies quality control:

- Normalize outputs to the five allowed labels.
- Drop invalid labels.
- Drop individual low-confidence LLM outputs when confidence scores are available.
- Majority vote across repeated labels for the same tweet.
- Filter low-agreement tweets with `--min-agreement`.
- Require multiple votes with `--min-votes` when several LLM passes are available.

The final emotion training file is:

```text
results/week14_crisismmd/processed/emotion.csv
```

This keeps emotion detection publishable as a weak-supervision experiment rather than treating raw LLM outputs as unquestioned ground truth.

### Week 14 Training and Metrics

The text classifier script uses Hugging Face transformer models. The intended model for the final emotion classifier is:

```text
microsoft/deberta-v3-base
```

The same training script can also train the informativeness, humanitarian, and damage-severity tasks:

```powershell
python src\week14\week14_train_text_classifier.py --task-csv results\week14_crisismmd\processed\humanitarian.csv --output-dir results\week14_crisismmd\humanitarian_deberta --loss focal
```

The primary metric is macro F1, because the humanitarian and emotion tasks are imbalanced and minority classes are operationally important. Week 14 also reports weighted F1, accuracy, and a confusion matrix.

The correct evaluation priority is:

```text
1. Macro F1
2. Weighted F1
3. Confusion matrix
```

### Week 14 Experimental Results

The first Week 14 experiments trained `microsoft/deberta-v3-base` on event-held-out CrisisMMD v2 splits. The reported result for each task is the best validation macro F1 epoch, because macro F1 is the primary metric under class imbalance.

```text
Task                  Best epoch   Accuracy   Macro F1   Weighted F1
Informativeness       2            0.7973     0.7032     0.7852
Humanitarian          1            0.6721     0.4708     0.6630
Damage severity       3            0.8424     0.3324     0.7860
```

The informativeness classifier is the strongest Week 14 baseline. This is expected because it is a binary task with substantially more stable textual cues.

The humanitarian classifier is harder because it is an eight-class task with severe class imbalance. The event-held-out validation split contains very small minority classes such as `vehicle_damage` and `missing_or_found_people`, so macro F1 is much lower than weighted F1. The best humanitarian run used weighted cross-entropy with square-root class weighting rather than aggressive focal loss.

The damage-severity proxy task reaches high accuracy and weighted F1, but macro F1 remains low. This suggests the model is learning the dominant severity pattern while still struggling with minority severity classes. This is an important limitation because the `image_damage` label is used as a text-only proxy target rather than a direct text annotation.

The first focal-loss humanitarian run collapsed because the smallest classes were extremely rare:

```text
missing_or_found_people: 23 training examples
vehicle_damage:         52 training examples
```

The final trainer therefore uses gentler class weighting by default and saves the best validation macro-F1 checkpoint:

```text
results/week14_crisismmd/<experiment_name>/best_checkpoint
```

### Week 14 Full Pipeline Integration

The social-media module produces a structured crisis understanding vector:

```text
Tweet Text
   |
   v
RoBERTa / DeBERTa Encoder
   |
   +-- Task 1: Informativeness
   +-- Task 2: Humanitarian Need
   +-- Task 3: Damage Severity Proxy
   +-- Task 4: Emotion
   |
   v
Structured Crisis Understanding Vector
   |
   v
Fusion with Satellite Damage + Topology Signals
   |
   v
RAG + LLM Report Generator
```

The main research contribution is that the final system is not only a damage classifier. It becomes a multi-source crisis understanding system combining:

- satellite vision for physical damage severity
- topology-guided validation for structural patterns
- social-media humanitarian classification
- emotion-aware crisis reasoning
- LLM-based report generation

This makes Week 14 an important bridge from image-based damage detection toward practical disaster-response intelligence.

## Future Extensions

After Week 14, the next improvements should focus on deployment realism and richer multi-source evaluation:

- Replace ground-truth object masks with predicted segmentation masks to test deployment realism.
- Add disaster-type performance breakdowns for each final model.
- Use embedding plots to identify whether minor/major errors are separability failures or label-quality failures.
- Investigate minor-damage label quality and visual ambiguity through crop-level error analysis.
- Add larger contextual crops, adaptive crop scaling, or oriented bounding boxes for subtle damage cues near building boundaries.
- Explore patch-level temporal transformers only after the topology verifier has been compared against the gated ConvNeXt baseline.
- Evaluate the Week 14 classifiers under both official CrisisMMD splits and event-held-out splits.
- Add a RAG report-generation stage that consumes satellite predictions, topology summaries, humanitarian predictions, and emotion labels together.
