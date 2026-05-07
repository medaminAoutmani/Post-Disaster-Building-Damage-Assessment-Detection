# Damage Detection Project

Satellite-image building segmentation project using the xBD disaster dataset.

The project builds a preprocessing and training pipeline for binary building segmentation from paired pre-disaster and post-disaster satellite images.

## Structure

```text
src/
+-- week1_preprocessing.py
+-- week2_dataset.py
+-- week2_model.py
+-- week2_train_baseline.py
+-- week3_dataset_statistics.py
+-- week3_model.py
+-- week3_train.py
+-- week4_model.py
+-- week4_train.py
splits/
+-- train.txt
+-- val.txt
+-- test.txt
results/
+-- week1/
+-- week2/
+-- week3/
+-- week4/
```

The raw dataset is intentionally ignored by Git with `.gitignore`.

## Stages

- Week 1: preprocessing, polygon parsing, mask generation, and visualization
- Week 2: first U-Net baseline training pipeline
- Week 3: improved baseline with dataset cleaning, BCE + Dice loss, more metrics, prediction visualization, and overfit testing
- Week 4: U-Net with a pretrained ResNet34 encoder for stronger feature extraction

See `PROJECT_REPORT.md` for the full project report.

## Commands

Create dataset statistics:

```powershell
python src\week3_dataset_statistics.py --data-dir data --split train --output-file results\week3\metrics\dataset_statistics.csv
```

Train Week 2 baseline:

```powershell
python src\week2_train_baseline.py --epochs 5 --batch-size 4 --image-size 512
```

Train Week 3 improved baseline:

```powershell
python src\week3_train.py --epochs 20 --batch-size 4 --image-size 512
```

Train Week 4 ResNet34 encoder U-Net:

```powershell
python src\week4_train.py --epochs 20 --batch-size 4 --image-size 512
```

By default, experiment artifacts are saved under:

```text
results/
```

This keeps Week 1 visualizations, Week 2 baseline checkpoints, Week 3 improved baseline outputs, and Week 4 pretrained-encoder outputs together in one results tree.

Run Week 3 overfit test:

```powershell
python src\week3_train.py --overfit-samples 8 --epochs 50 --batch-size 2 --small-model
```
