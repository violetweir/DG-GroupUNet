# Swin-UNet

This repository includes a separated Swin-UNet integration for polyp segmentation.

The Swin-UNet files are intentionally kept separate from the main training and testing entry points because the model requires a fixed `img_size` and should be initialized from the pretrained Swin-T checkpoint.

## Files

```text
networks/swinunet/
networks/swinunet_network.py
tools/train_swinunet_polyp.py
tools/test_swinunet_polyp.py
```

The main scripts below are not modified to import Swin-UNet:

```text
tools/train_polyp.py
tools/test_polyp.py
```

## Source

The model code is adapted from:

```text
https://github.com/HUCAOFIGHTING/SWIN-UNET
```

Checked source commit:

```text
f48f623
```

## Environment

Install the normal project dependencies plus `einops`:

```bash
pip install einops
```

The upstream repository recommends Python 3.7, but the Swin-UNet model itself does not require CUDA-specific custom extensions like VM-UNet.

## Pretrained Checkpoint

Training requires a pretrained Swin-T checkpoint. Download the upstream pretrained model and place it under:

```text
pretrained_ckpt/swin_tiny_patch4_window7_224.pth
```

The training script requires `--pretrained_ckpt` and will stop if the file does not exist.

## Data Layout

The scripts follow the existing polyp dataset layout:

```text
data/polyp/target/ClinicDB/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

## Training

From the repository root:

```bash
python tools/train_swinunet_polyp.py \
  --pretrained_ckpt pretrained_ckpt/swin_tiny_patch4_window7_224.pth
```

Common options:

```bash
python tools/train_swinunet_polyp.py \
  --pretrained_ckpt pretrained_ckpt/swin_tiny_patch4_window7_224.pth \
  --epoch 200 \
  --batchsize 4 \
  --img_size 352
```

Swin-UNet uses fixed input size internally, so this dedicated script does not use multi-scale training.

Training outputs are saved together:

```text
model_pth/<run_id>/
  <run_id>-last.pth
  <run_id>-best.pth
  train_log_<run_id>.log
```

## Testing

Use the `run_id` printed during training:

```bash
python tools/test_swinunet_polyp.py --run_id <run_id>
```

Optional arguments:

```bash
python tools/test_swinunet_polyp.py --run_id <run_id> --dataset_name ClinicDB --split test --img_size 352
```

Prediction masks and result spreadsheets are saved to:

```text
predictions_polyp/<run_id>/<dataset_name>/<split>/
results_polyp/Results_<run_id>_<dataset_name>_<split>.xlsx
All_Runs_Summary_Polyp.xlsx
```

## Notes

- The model is constructed with `num_classes=1` for binary polyp segmentation.
- The model returns logits and training uses `binary_cross_entropy_with_logits`.
- Testing applies sigmoid during prediction post-processing.
- Pretraining is required by the dedicated training script because Swin-UNet is sensitive to initialization.
