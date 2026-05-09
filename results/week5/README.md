# Week 5 Results

Week 5 artifacts for multiclass xBD damage segmentation.

## Objective

Move from binary building segmentation to 5-class damage segmentation:

| Class ID | Meaning |
|---:|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

## Model

- Encoder: ResNet34
- Input: 6 channels, pre-disaster RGB plus post-disaster RGB
- Output: 5 channels, one logit map per class
- Loss: weighted CrossEntropy plus multiclass Dice
- Important metrics: per-class Dice, mean foreground Dice, macro F1, pixel accuracy, IoU, and confusion matrix

## Contents

- `checkpoints/`: best Week 5 multiclass model weights
- `config/`: training configuration
- `metrics/`: final metrics, training log, dataset statistics, and prediction records
- `predictions/`: qualitative color-mask prediction examples
- `visualizations/`: curves, color masks, overlays, and comparison panels
- `confusion_matrices/`: CSV and heatmap confusion matrices

Train Week 5 with:

```powershell
python src\week5_train.py --epochs 20 --batch-size 4 --image-size 512
```

Run a small overfit check with:

```powershell
python src\week5_train.py --overfit-samples 8 --epochs 50 --batch-size 2
```

Run without class weights:

```powershell
python src\week5_train.py --no-class-weights
```

Run a frozen-encoder experiment:

```powershell
python src\week5_train.py --freeze-encoder
```
