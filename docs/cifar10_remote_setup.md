# CIFAR-10 DDPM Remote Training

This repo's `Trainer` reads images from a folder, so CIFAR-10 is first exported
to PNG files and then trained with `train_cifar10.py`.

## 1. Server Setup

Check the GPU:

```bash
nvidia-smi
```

Create an environment:

```bash
conda create -n ddpm-cifar python=3.10 -y
conda activate ddpm-cifar
python --version
```

Install PyTorch. CUDA 12.8 is a good default for current NVIDIA servers:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If the driver is too old for CUDA 12.8, use the command recommended by the
official PyTorch selector: https://pytorch.org/get-started/locally/

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Install this repo:

```bash
cd /path/to/denoising-diffusion-pytorch
pip install -e .
```

Optional:

```bash
pip install matplotlib tensorboard
```

## 2. Prepare CIFAR-10

```bash
python scripts/prepare_cifar10_folder.py --root data/cifar10 --split train
```

Expected output:

```text
data/cifar10/
  raw/
  train/
    airplane_000000.png
    automobile_000001.png
    ...
```

For a quick export test:

```bash
python scripts/prepare_cifar10_folder.py --root data/cifar10 --split train --limit 1000
```

## 3. Smoke Test

```bash
python train_cifar10.py \
  --train-num-steps 100 \
  --save-and-sample-every 50 \
  --batch-size 32
```

Check:

```bash
ls results/cifar10
```

You should see `sample-*.png` and `model-*.pt`.

## 4. Full Training

```bash
mkdir -p logs
nohup python train_cifar10.py \
  --train-num-steps 100000 \
  --save-and-sample-every 1000 \
  --batch-size 64 \
  --results-folder results/cifar10 \
  > logs/cifar10.log 2>&1 &
```

Watch logs:

```bash
tail -f logs/cifar10.log
```

Multi-GPU:

```bash
accelerate config
accelerate launch train_cifar10.py \
  --train-num-steps 100000 \
  --save-and-sample-every 1000 \
  --batch-size 64 \
  --results-folder results/cifar10
```

## 5. Resume and FID

Resume from `results/cifar10/model-50.pt`:

```bash
python train_cifar10.py --resume 50
```

Enable FID later, after samples become meaningful:

```bash
python train_cifar10.py \
  --train-num-steps 300000 \
  --calculate-fid \
  --num-fid-samples 10000
```

## What to Expect

- 1k steps: mostly noise or blurry color blocks.
- 10k steps: coarse colors and shapes.
- 50k steps: visible CIFAR-like structure.
- 100k+ steps: clearer objects, though not necessarily paper-level quality.
