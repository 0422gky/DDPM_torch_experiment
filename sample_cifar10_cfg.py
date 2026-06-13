import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from train_cifar10_cfg import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a CIFAR-10 classifier-free guidance checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_cfg/sample_custom.png"))
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--cond-scale", type=float, default=4.0)
    parser.add_argument("--rescaled-phi", type=float, default=0.7)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--cond-drop-prob", type=float, default=0.2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=250)
    parser.add_argument("--objective", choices=("pred_noise", "pred_x0", "pred_v"), default="pred_v")
    parser.add_argument("--beta-schedule", choices=("linear", "cosine"), default="cosine")
    return parser.parse_args()


def main():
    args = parse_args()

    from ema_pytorch import EMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = build_model(args).to(device)
    data = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ema = EMA(diffusion, beta=0.995, update_every=10)
    ema.load_state_dict(data["ema"])
    ema.ema_model.to(device)
    ema.ema_model.eval()

    classes = torch.tensor(args.classes, device=device, dtype=torch.long)
    classes = classes.repeat_interleave(args.samples_per_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        samples = ema.ema_model.sample(
            classes=classes,
            cond_scale=args.cond_scale,
            rescaled_phi=args.rescaled_phi,
        )

    save_image(samples, args.output, nrow=args.samples_per_class)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
