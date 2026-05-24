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
+-- week5_model.py
+-- week5_train.py
+-- week6/
    +-- week6_losses.py
    +-- week6_sampler.py
    +-- week6_metrics.py
    +-- week6_visualization.py
    +-- week6_analysis.py
    +-- week6_ablation.py
    +-- week6_scheduler.py
    +-- week6_utils.py
    +-- week6_augmentations.py
    +-- week6_model_attention_unet.py
    +-- week6_model_unetplusplus.py
    +-- week6_model_deeplabv3plus.py
    +-- week6_model_resnet50_unet.py
    +-- week6_train_attention_unet.py
    +-- week6_train_unetplusplus.py
    +-- week6_train_deeplabv3plus.py
    +-- week6_train_resnet50_unet.py
    +-- week6_experiment_runner.py
    +-- week6_inference.py
+-- week7/
    +-- week7_experiment_runner.py
    +-- week7_dataset.py
    +-- week7_temporal_fusion.py
    +-- week7_attention.py
    +-- week7_losses.py
    +-- week7_metrics.py
    +-- week7_visualization.py
    +-- week7_analysis.py
    +-- week7_utils.py
    +-- week7_callbacks.py
    +-- week7_model_siamese_resnet50_unet.py
    +-- week7_model_siamese_attention_unet.py
    +-- week7_model_siamese_deeplab.py
    +-- week7_train_siamese_baseline.py
    +-- week7_train_siamese_attention.py
    +-- week7_train_cbam.py
    +-- week7_train_nonlocal.py
    +-- week7_inference.py
splits/
+-- train.txt
+-- val.txt
+-- test.txt
results/
+-- week1/
+-- week2/
+-- week3/
+-- week4/
+-- week5/
+-- week6/
+-- week7/
```

The raw dataset is intentionally ignored by Git with `.gitignore`.

## Stages

- Week 1: preprocessing, polygon parsing, mask generation, and visualization
- Week 2: first U-Net baseline training pipeline
- Week 3: improved baseline with dataset cleaning, BCE + Dice loss, more metrics, prediction visualization, and overfit testing
- Week 4: U-Net with a pretrained ResNet34 encoder for stronger feature extraction
- Week 5: multiclass damage segmentation for background, no damage, minor damage, major damage, and destroyed
- Week 6: research experiments with isolated runs, ablations, advanced losses, samplers, metrics, and upgraded architectures
- Week 7: temporal attention-based Siamese damage segmentation with fusion and attention ablations

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

Train Week 5 multiclass damage segmentation:

```powershell
python src\week5_train.py --epochs 20 --batch-size 4 --image-size 512
```

Create the Week 6 research result tree:

```powershell
python src\week6\week6_experiment_runner.py --scaffold-only
```

Run Week 6 experiments:

```powershell
.\run_week6_experiments.ps1
```

`week6_train_deeplabv3plus.py` runs a torchvision DeepLabV3 model; the filename is kept for continuity, but the saved metadata uses the correct DeepLabV3 architecture name.

The default Week 6 script is now Morocco-focused. It trains only on sample IDs containing earthquake, flood/flooding, wildfire, or fire, and runs this faster order:

```text
baseline -> focal_loss -> weighted_sampler -> attention_unet -> resnet50
```

Run the full research suite only when you have time:

```powershell
.\run_week6_experiments.ps1 -FullSuite
```

Run one Morocco-adaptation experiment manually:

```powershell
python src\week6\week6_experiment_runner.py --experiment attention_unet --morocco-adaptation --epochs 12
```

Recommended research order:

```text
1. Baseline reproduction
2. Ablation study: focal loss, weighted sampler, attention U-Net, ResNet50
3. Cheap Optuna HPO on the strongest candidate
4. Final comparison table and final model training
```

Run cheap Optuna HPO for Morocco adaptation:

```powershell
.\run_week6_experiments.ps1 -HPO -Trials 12
```

Or run it directly:

```powershell
python src\week6\week6_experiment_runner.py --experiment attention_unet --morocco-adaptation --optuna-study --optuna-trials 12 --optuna-epochs 6 --optuna-max-train-samples 2000 --optuna-max-val-samples 500
```

Optuna writes trial folders under `results/week6/experiment_attention_unet_optuna_trial_*` and study summaries under `results/week6/hyperparameter_optimization/morocco_week6_hpo/`.

Run a multi-seed Week 6 experiment:

```powershell
python src\week6\week6_experiment_runner.py --experiment attention_unet --seeds 42 123 999
```

Run k-fold validation:

```powershell
python src\week6\week6_experiment_runner.py --experiment attention_unet --k-folds 5 --seeds 42 123
```

Week 6 training saves TensorBoard logs, early-stopping metadata, AMP settings, environment metadata, precision-recall CSVs, best/failure/random validation panels, confidence maps, error heatmaps, and overlays inside each isolated experiment folder.

Run standalone inference from a Week 6 checkpoint:

```powershell
python src\week6\week6_inference.py --checkpoint results\week6\experiment_attention_unet\checkpoints\best_model.pt --model attention_unet
```

Refresh Week 6 comparative summaries:

```powershell
python src\week6\week6_experiment_runner.py --summarize-only
```

Create the Week 7 temporal Siamese result tree:

```powershell
python src\week7\week7_experiment_runner.py --scaffold-only
```

Recommended Week 7 order:

```text
siamese_concat -> siamese_difference -> siamese_bottleneck_attention -> siamese_cbam -> siamese_nonlocal
```

Week 7 Phase 1 uses the best Week 6 ResNet50-UNet hyperparameters with separate Siamese learning-rate groups:

```text
encoder_lr=0.00028325734588543713
fusion_lr=0.0003
decoder_lr=0.0008129702163604196
loss=cross_entropy_dice
sampler=weighted
scheduler=reduce_on_plateau
batch_size=4
class_weights=1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
```

Run Week 7 temporal baselines:

```powershell
python src\week7\week7_experiment_runner.py --experiment siamese_concat --morocco-adaptation --epochs 20 --batch-size 4 --encoder-lr 0.00028325734588543713 --fusion-lr 0.0003 --decoder-lr 0.0008129702163604196 --loss cross_entropy_dice --sampler weighted --scheduler reduce_on_plateau --class-weights 1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
python src\week7\week7_experiment_runner.py --experiment siamese_difference --morocco-adaptation --epochs 20 --batch-size 4 --encoder-lr 0.00028325734588543713 --fusion-lr 0.0003 --decoder-lr 0.0008129702163604196 --loss cross_entropy_dice --sampler weighted --scheduler reduce_on_plateau --class-weights 1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
```

Run Week 7 attention experiments:

```powershell
python src\week7\week7_experiment_runner.py --experiment siamese_bottleneck_attention --morocco-adaptation --epochs 20 --batch-size 4 --encoder-lr 0.00028325734588543713 --fusion-lr 0.0003 --decoder-lr 0.0008129702163604196 --loss cross_entropy_dice --sampler weighted --scheduler reduce_on_plateau --class-weights 1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
python src\week7\week7_experiment_runner.py --experiment siamese_cbam --morocco-adaptation --epochs 20 --batch-size 4 --encoder-lr 0.00028325734588543713 --fusion-lr 0.0003 --decoder-lr 0.0008129702163604196 --loss cross_entropy_dice --sampler weighted --scheduler reduce_on_plateau --class-weights 1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
python src\week7\week7_experiment_runner.py --experiment siamese_nonlocal --morocco-adaptation --epochs 20 --batch-size 4 --encoder-lr 0.00028325734588543713 --fusion-lr 0.0003 --decoder-lr 0.0008129702163604196 --loss cross_entropy_dice --sampler weighted --scheduler reduce_on_plateau --class-weights 1.0 0.2 1.4323335332729514 2.817887932741608 2.301194607934551
```

Run Week 7 inference:

```powershell
python src\week7\week7_inference.py --checkpoint results\week7\experiment_siamese_concat\checkpoints\best_model.pt --fusion concat --attention no_attention
```

By default, experiment artifacts are saved under:

```text
results/
```

This keeps Week 1 visualizations, Week 2 baseline checkpoints, Week 3 improved baseline outputs, Week 4 pretrained-encoder outputs, and Week 5 multiclass damage outputs together in one results tree.

Week 6 experiments are isolated under descriptive folders such as `results/week6/experiment_attention_unet/` and `results/week6/experiment_resnet50/` so ablation runs do not overwrite each other.

Run Week 3 overfit test:

```powershell
python src\week3_train.py --overfit-samples 8 --epochs 50 --batch-size 2 --small-model
```
