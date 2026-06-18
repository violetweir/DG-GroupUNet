# UltraLight VM-UNet

This repository includes a separated UltraLight VM-UNet integration for polyp segmentation.

The VM-UNet files are intentionally kept separate from the main training and testing entry points because this model depends on `mamba_ssm`, `causal_conv1d`, and `triton`, which are sensitive to Python, PyTorch, CUDA, and wheel versions.

## Files

```text
networks/vmunet_network.py
tools/train_vmunet_polyp.py
tools/test_vmunet_polyp.py
```

The main scripts below are not modified to import VM-UNet:

```text
tools/train_polyp.py
tools/test_polyp.py
```

## Environment

Use a dedicated conda environment for VM-UNet. Do not install these dependencies into the main project environment unless you are sure the versions are compatible.

```bash
conda create -n vmunet python=3.8
conda activate vmunet

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
pip install loguru tqdm pyyaml pandas opencv-python seaborn albumentations==1.1.0 tabulate pillow openpyxl
```

If `causal_conv1d` or `mamba_ssm` fails to install from pip, use wheels that match your CUDA, PyTorch, Python, and ABI exactly.

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
python tools/train_vmunet_polyp.py
```

Common options:

```bash
python tools/train_vmunet_polyp.py --epoch 200 --batchsize 8 --img_size 352
```

Disable multi-scale training:

```bash
python tools/train_vmunet_polyp.py --no_multiscale
```

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
python tools/test_vmunet_polyp.py --run_id <run_id>
```

Optional arguments:

```bash
python tools/test_vmunet_polyp.py --run_id <run_id> --dataset_name ClinicDB --split test --img_size 352
```

Prediction masks and result spreadsheets are saved to:

```text
predictions_polyp/<run_id>/<dataset_name>/<split>/
results_polyp/Results_<run_id>_<dataset_name>_<split>.xlsx
All_Runs_Summary_Polyp.xlsx
```

## Notes

- `networks/vmunet_network.py` safely imports without `mamba_ssm`, but model creation requires `mamba_ssm`.
- The model returns logits as `[out0]`, not sigmoid probabilities, so it works with `binary_cross_entropy_with_logits`.
- Testing applies sigmoid during prediction post-processing.
- Keep VM-UNet experiments in the dedicated `vmunet` environment to avoid breaking the main MK-UNet environment.
