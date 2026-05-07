# Damage Detection Project Report

## Project Description

This project builds a satellite-image damage detection pipeline using the xBD disaster dataset. The main goal is to identify building regions from paired pre-disaster and post-disaster satellite images, then prepare the system for reliable segmentation-based disaster damage analysis.

The dataset contains:

- Pre-disaster RGB satellite images
- Post-disaster RGB satellite images
- JSON label files containing building polygons
- Damage labels such as `no-damage`, `minor-damage`, `major-damage`, `destroyed`, and `un-classified`

The current project focuses on binary building segmentation. This means the model learns to predict whether each pixel belongs to a labeled building or background. The pre-disaster and post-disaster images are stacked together as a 6-channel input, allowing the model to use both before-and-after visual information.

The full pipeline is divided into three stages:

1. Week 1: Data exploration, preprocessing, visualization, and mask generation
2. Week 2: Dataset pipeline, train/validation/test splits, and the first baseline U-Net
3. Week 3: A separate improved baseline with cleaner data, stronger training, prediction visualization, overfit testing, and better metrics

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
outputs/visualizations/
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
outputs/checkpoints/week2_unet_binary_best.pt
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
| Week 2 baseline | `src/week2_model.py` | `src/week2_train_baseline.py` | `outputs/checkpoints/week2_unet_binary_best.pt` |
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

It saves a report-friendly CSV file to:

```text
outputs/logs/week3_train_dataset_statistics.csv
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
results/week3/predictions/epoch_XXX/
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

The project now follows this output structure:

```text
outputs/
+-- checkpoints/
+-- logs/
+-- masks/
+-- predictions/
+-- visualizations/
results/
+-- week3/
```

Each folder has a clear purpose:

| Folder | Purpose |
|---|---|
| `outputs/checkpoints/` | Saved model weights |
| `outputs/logs/` | Dataset statistics and experiment logs |
| `outputs/masks/` | Generated or saved mask artifacts |
| `outputs/predictions/` | Model prediction visualizations |
| `outputs/visualizations/` | Week 1 sample visualizations |
| `results/week3/` | Reproducibility package for Week 3 metrics, logs, curves, panels, checkpoints, dataset statistics, and failure analysis |

## How to Run the Project

Create dataset statistics:

```powershell
python src\week3_dataset_statistics.py --data-dir data --split train --output-dir outputs\logs
```

Train the Week 2 baseline:

```powershell
python src\week2_train_baseline.py --epochs 5 --batch-size 4 --image-size 512
```

Train the Week 3 improved baseline:

```powershell
python src\week3_train.py --epochs 20 --batch-size 4 --image-size 512
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
- An overfit test is available for debugging the full pipeline.

## Next Steps

The next improvement would be to move from binary building segmentation toward multiclass damage segmentation. Instead of predicting only building versus background, the model could predict:

- background
- no-damage
- minor-damage
- major-damage
- destroyed

Other useful future improvements include testing larger U-Net variants, using pretrained encoders, tracking experiments in a CSV log, adding confusion-matrix analysis, and comparing performance across disaster types.
