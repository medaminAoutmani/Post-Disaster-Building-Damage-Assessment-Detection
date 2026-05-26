# Damage Detection Project Report

## Project Description

This project builds a satellite-image damage detection pipeline using the xBD disaster dataset. The main goal is to identify building regions from paired pre-disaster and post-disaster satellite images, then prepare the system for reliable segmentation-based disaster damage analysis.

The dataset contains:

- Pre-disaster RGB satellite images
- Post-disaster RGB satellite images
- JSON label files containing building polygons
- Damage labels such as `no-damage`, `minor-damage`, `major-damage`, `destroyed`, and `un-classified`

The project begins with binary building segmentation, where the model learns whether each pixel belongs to a labeled building or background. It then extends the same pipeline into multiclass damage segmentation, temporal Siamese modeling, class-imbalance handling, and multi-task learning. In the early models, the pre-disaster and post-disaster images are stacked together as a 6-channel input. In the later Siamese models, the two images are processed as separate RGB streams and fused at the feature level.

The current pipeline is divided into ten development stages:

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

## Future Extensions

After the Week 10 building-aware loss experiments, the next improvements should focus on making damage severity prediction more reliable:

- Treat the plain full-image CrossEntropy damage objective as the current stronger baseline.
- If soft building weighting is tested again, use milder background weights such as 0.5 or 0.7 instead of 0.2.
- Use Week 8 balanced training splits to test whether extra minority samples improve minor-damage Dice.
- Tune the balance between building auxiliary losses and damage loss.
- Compare focal loss, moderate class-weighted CrossEntropy, and rare-class oversampling under the same Week 9 architecture.
- Add disaster-type performance breakdowns for each final model.
- Add confusion-matrix discussion for minor/major/destroyed mistakes.
- Investigate crop-based high-resolution training for tiny buildings.
- Test larger encoders or transformer-based segmentation backbones if GPU memory allows.
