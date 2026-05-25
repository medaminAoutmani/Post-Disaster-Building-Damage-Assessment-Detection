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
+-- week8/
    +-- week8_class_distribution.py
    +-- week8_prune_dataset.py
    +-- week8_select_minority_samples.py
    +-- week8_prepare_balanced_data.py
+-- week9/
    +-- week9_dataset.py
    +-- week9_model_multitask_siamese.py
    +-- week9_losses.py
    +-- week9_metrics.py
    +-- week9_train_multitask.py
+-- week10/
    +-- week10_train_masked_loss.py
    +-- week10_train_soft_masked_loss.py
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
+-- week8/
+-- week9/
+-- week10/
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
- Week 8: rare-class imbalance audit and targeted data expansion, especially for minor damage
- Week 9: multi-task Siamese damage segmentation with auxiliary building-mask supervision
- Week 10: building-masked damage loss for rare-class learning on building pixels only

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

Week 8 rare-class imbalance task:

```text
Goal:
Audit the current Morocco-focused train/val data before downloading extra xBD metadata parts.

Current issue:
Week 7 models learn background, no_damage, and destroyed reasonably well, but minor_damage remains near zero.

Step 1 - Compute current class distribution:
- pixels per class for train and val
- images containing each class for train and val
- buildings/polygons per class for train and val
- per-disaster class distribution for earthquake, flood/flooding, wildfire/fire

Step 2 - Decide what extra data is needed:
- prioritize minor_damage samples first
- add major_damage samples only if major remains weak
- add destroyed samples only if needed, because destroyed is already learned better

Step 3 - Download targeted xBD full-data metadata parts:
- inspect metadata before adding images
- keep only samples that contain useful minority-class buildings/pixels
- prefer earthquake, flood/flooding, wildfire/fire to stay close to the Morocco adaptation target

Step 4 - Preserve fair evaluation:
- add extra samples only to the training split
- keep validation and test unchanged
- avoid adding train samples from scenes/disasters that leak into val/test

Expected output:
- results/week8/class_distribution_train_val.csv
- results/week8/per_disaster_class_distribution.csv
- results/week8/selected_extra_minority_samples.csv
- updated training split for the balanced Week 8 experiment
```

Find non-Morocco or invalid samples that can be removed to save storage:

```powershell
py src\week8\week8_prune_dataset.py --mode both
```

This writes `results/week8/deletion_candidates.csv` and does not delete files unless `--delete --confirm-delete DELETE_NON_MOROCCO_DATA` are both provided.

After downloading an extra xBD full-data part, rank minority-rich samples before merging anything:

```powershell
py src\week8\week8_select_minority_samples.py --candidate-data-dir path\to\downloaded_xbd_part --morocco-adaptation --min-minor-buildings 1 --min-minority-pixels 100 --top-k 500
```

This writes `results/week8/selected_extra_minority_samples.csv`. Prioritize samples with many `minor_damage` and `major_damage` buildings; keep validation and test unchanged.

Copy selected samples into a separate project folder, create a balanced Week 8 train split, and compare before/after minority counts:

```powershell
py src\week8\week8_prepare_balanced_data.py --selected-csv results\week8\selected_extra_minority_samples.csv --top-k 500
```

This creates:

```text
data/week8_extra/train/images/
data/week8_extra/train/labels/
splits/week8_train_balanced.txt
results/week8/copied_extra_minority_samples.csv
results/week8/class_distribution_week8_before_after.csv
results/week8/minority_before_after.csv
```

The original `splits/train.txt`, `splits/val.txt`, and `splits/test.txt` are not modified.

Week 9 multi-task Siamese roadmap:

```text
Goal:
Train a multi-task Siamese ResNet50-UNet that jointly predicts:
- pre-disaster building mask
- post-disaster building mask
- 5-class damage mask

Best Week 7 starting point:
Siamese ResNet50-UNet + difference fusion + CBAM

Week 9 additions:
- shared ResNet50 encoder
- auxiliary pre/post building heads
- damage classification head
- multi-task loss: pre building + post building + 3 * damage
- warmup cosine scheduler
- gradient clipping
- stronger rare-class weights
- optional Week 8 selected extra training samples
```

Run Week 9A multi-task architecture without attention, using the old training set only:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_difference --morocco-adaptation --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 8 12 12 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Run Week 9B multi-task CBAM with the old training set only:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --morocco-adaptation --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 8 12 12 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Run Week 9C multi-task CBAM with the old training set plus selected Week 8 extra samples:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --experiment-name experiment_multitask_cbam_difference_week8_extra --morocco-adaptation --use-week8-extra --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 8 12 12 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Run Week 9D loss-balancing ablation after Week 9B/9C:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --experiment-name experiment_multitask_cbam_lambda_damage_1 --morocco-adaptation --use-week8-extra --epochs 50 --lambda-damage 1
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --experiment-name experiment_multitask_cbam_lambda_damage_2 --morocco-adaptation --use-week8-extra --epochs 50 --lambda-damage 2
python src\week9\week9_train_multitask.py --experiment multitask_cbam_difference --experiment-name experiment_multitask_cbam_lambda_damage_5 --morocco-adaptation --use-week8-extra --epochs 50 --lambda-damage 5
```

Retry non-local attention only after CBAM is stable:

```powershell
python src\week9\week9_train_multitask.py --experiment multitask_nonlocal_difference --morocco-adaptation --use-week8-extra --epochs 50 --batch-size 2 --encoder-lr 0.00005 --fusion-lr 0.0001 --decoder-lr 0.0002 --no-amp --grad-clip-norm 1.0
```

Week 10A building-masked damage loss:

```text
Goal:
Keep the Week 9 multi-task CBAM architecture fixed, but compute damage loss only on ground-truth building pixels.

Hypothesis:
Damage labels are meaningful only where damage_mask > 0. Masking background pixels during damage CE+Dice should reduce background dominance and improve minor/rare-class learning.

Fixed architecture:
- multi-task Siamese ResNet50-UNet
- difference fusion
- CBAM
- pre/post building heads
- damage head

Only changed component:
- damage loss: building_masked_ce_dice
```

Run Week 10A-1 using old/current training data only:

```powershell
python src\week10\week10_train_masked_loss.py --experiment multitask_cbam_difference --morocco-adaptation --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 6 10 10 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Run Week 10A-2 using top150 Week 8 extra samples:

```powershell
python src\week10\week10_train_masked_loss.py --experiment multitask_cbam_difference --experiment-name experiment_week10a_masked_loss_week8_extra_top150 --morocco-adaptation --use-week8-extra --max-extra-samples 150 --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 6 10 10 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Week 10A.1 soft building-weighted damage loss:

```text
Rationale:
The hard masked loss removes almost all background supervision. Soft weighting keeps background pixels in the objective with lower weight while emphasizing building pixels.

Damage pixel weights:
- background pixels: 0.2
- building pixels: 1.0

Loss:
soft weighted CE + soft weighted Dice on damage classes
```

Recommended Week 10A.1 run using old/current training data only:

```powershell
python src\week10\week10_train_soft_masked_loss.py --experiment multitask_cbam_difference --morocco-adaptation --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 6 10 10 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
```

Only run the top150 extra version after the old-data soft loss is stable:

```powershell
python src\week10\week10_train_soft_masked_loss.py --experiment multitask_cbam_difference --experiment-name experiment_week10a1_soft_masked_loss_week8_extra_top150 --morocco-adaptation --use-week8-extra --max-extra-samples 150 --epochs 50 --batch-size 4 --encoder-lr 0.0001 --fusion-lr 0.0003 --decoder-lr 0.0005 --scheduler warmup_cosine --damage-class-weights 1 2 6 10 10 --lambda-pre 1 --lambda-post 1 --lambda-damage 3 --grad-clip-norm 1.0
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
