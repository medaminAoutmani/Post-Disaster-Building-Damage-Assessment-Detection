# Week 4 Results

Week 4 artifacts for the U-Net model with a ResNet34 encoder.

## Model

- Encoder: ResNet34
- Input: 6 channels, pre-disaster RGB plus post-disaster RGB
- Output: binary building segmentation mask
- Initialization: ImageNet pretrained ResNet34 weights by default
- Normalization: ImageNet mean and standard deviation duplicated for pre/post RGB

The first ResNet convolution is adapted from 3 input channels to 6 input channels by splitting the pretrained RGB weights across the pre-disaster and post-disaster image channels.

## Contents

- `checkpoints/`: best Week 4 model weights
- `config/`: training configuration
- `metrics/`: final metrics, training logs, dataset statistics, and prediction records
- `predictions/`: categorized qualitative prediction examples
- `visualizations/`: curves, masks, overlays, and comparison panels

Train Week 4 with:

```powershell
python src\week4_train.py --epochs 20 --batch-size 4 --image-size 512
```

Run a small overfit check with:

```powershell
python src\week4_train.py --overfit-samples 8 --epochs 50 --batch-size 2
```
