# CIFAR-10 Guidance Experiments

This repo now has two CIFAR-10 guidance paths:

- Classifier-free guidance: train one class-conditional diffusion model with condition dropout.
- Classifier guidance: reuse an unconditional diffusion model, train a noisy classifier, then guide sampling with classifier gradients.

For a Tesla P4 8GB GPU, keep batch sizes small and avoid flash attention.

## Classifier-Free Guidance

Train a class-conditional diffusion model:

```bash
mkdir -p logs
nohup python train_cifar10_cfg.py \
  --data-root data/cifar10/raw \
  --results-folder results/cifar10_cfg \
  --batch-size 8 \
  --gradient-accumulate-every 8 \
  --train-num-steps 100000 \
  --save-and-sample-every 1000 \
  --samples-per-class 2 \
  --amp \
  > logs/cifar10_cfg.log 2>&1 &
```

During training, `sample-i.png` contains class-ordered samples. With
`--samples-per-class 2`, each CIFAR-10 class contributes two generated images.

Sample manually from a checkpoint:

```bash
python sample_cifar10_cfg.py \
  --checkpoint results/cifar10_cfg/model-100.pt \
  --output results/cifar10_cfg/sample_scale4.png \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --samples-per-class 4 \
  --cond-scale 4
```

Try `--cond-scale` values from `1` to `8`. Larger values usually improve class
fidelity but can reduce image diversity or add artifacts.

## Classifier Guidance

This path needs two trained models:

1. An unconditional diffusion checkpoint from `train_cifar10.py`.
2. A noisy classifier trained on `x_t` at random diffusion timesteps.

Train the noisy classifier:

```bash
mkdir -p logs
nohup python train_cifar10_noisy_classifier.py \
  --data-root data/cifar10/raw \
  --results-folder results/cifar10_classifier \
  --batch-size 128 \
  --train-num-steps 50000 \
  --save-every 5000 \
  > logs/cifar10_classifier.log 2>&1 &
```

If P4 memory is tight, lower `--batch-size` to `64` or `32`.

Sample with classifier guidance:

```bash
python sample_cifar10_classifier_guided.py \
  --diffusion-checkpoint results/cifar10/model-100.pt \
  --classifier-checkpoint results/cifar10_classifier/classifier-latest.pt \
  --output results/cifar10_classifier_guided/sample_scale2.png \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --samples-per-class 2 \
  --guidance-scale 2
```

Try `--guidance-scale` values from `0.5` to `4`. Too high a scale can create
sharp but unnatural samples.

## CIFAR-10 Class IDs

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

## Recommended Order

Start with classifier-free guidance. It is simpler, usually more stable, and
does not need a separate classifier. Then use classifier guidance to compare how
external classifier gradients change class fidelity and visual artifacts.
