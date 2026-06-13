import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm.auto import tqdm


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier-free guided diffusion.")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10/raw"))
    parser.add_argument("--results-folder", type=Path, default=Path("results/cifar10_cfg"))
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--cond-drop-prob", type=float, default=0.2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=250)
    parser.add_argument("--objective", choices=("pred_noise", "pred_x0", "pred_v"), default="pred_v")
    parser.add_argument("--beta-schedule", choices=("linear", "cosine"), default="cosine")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulate-every", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=8e-5)
    parser.add_argument("--train-num-steps", type=int, default=100000)
    parser.add_argument("--save-and-sample-every", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")

    parser.add_argument("--cond-scale", type=float, default=4.0)
    parser.add_argument("--rescaled-phi", type=float, default=0.7)
    parser.add_argument("--samples-per-class", type=int, default=4)
    return parser.parse_args()


def build_model(args):
    from denoising_diffusion_pytorch.classifier_free_guidance import (
        GaussianDiffusion,
        Unet,
    )

    model = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        num_classes=10,
        cond_drop_prob=args.cond_drop_prob,
        channels=3,
    )

    return GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
    )


def save_checkpoint(path, step, accelerator, model, opt, ema):
    data = {
        "step": step,
        "model": accelerator.get_state_dict(model),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
    }
    torch.save(data, path)


def load_checkpoint(path, accelerator, model, opt, ema):
    data = torch.load(path, map_location=accelerator.device, weights_only=True)
    accelerator.unwrap_model(model).load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])
    ema.load_state_dict(data["ema"])
    return int(data["step"])


def main():
    args = parse_args()

    from ema_pytorch import EMA

    args.results_folder.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision if args.amp else "no")
    device = accelerator.device

    transform = T.Compose([
        T.Resize(args.image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    dataset = CIFAR10(root=str(args.data_root), train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    dataloader = cycle(accelerator.prepare(dataloader))

    diffusion = build_model(args)
    opt = Adam(diffusion.parameters(), lr=args.train_lr, betas=(0.9, 0.99))

    diffusion, opt = accelerator.prepare(diffusion, opt)
    ema = EMA(accelerator.unwrap_model(diffusion), beta=args.ema_decay, update_every=10)
    ema.to(device)

    step = 0
    if args.resume is not None:
        step = load_checkpoint(args.results_folder / f"model-{args.resume}.pt", accelerator, diffusion, opt, ema)

    pbar = tqdm(initial=step, total=args.train_num_steps, disable=not accelerator.is_main_process)

    while step < args.train_num_steps:
        diffusion.train()
        total_loss = 0.0

        for _ in range(args.gradient_accumulate_every):
            images, labels = next(dataloader)
            images = images.to(device)
            labels = labels.to(device)

            with accelerator.autocast():
                loss = diffusion(images, classes=labels)
                loss = loss / args.gradient_accumulate_every
                total_loss += loss.item()

            accelerator.backward(loss)

        accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        step += 1
        pbar.set_description(f"loss: {total_loss:.4f}")
        pbar.update(1)

        if accelerator.is_main_process:
            ema.update()

            if step % args.save_and_sample_every == 0:
                milestone = step // args.save_and_sample_every
                ema.ema_model.eval()

                classes = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
                with torch.inference_mode():
                    samples = ema.ema_model.sample(
                        classes=classes,
                        cond_scale=args.cond_scale,
                        rescaled_phi=args.rescaled_phi,
                    )

                nrow = args.samples_per_class
                save_image(samples, args.results_folder / f"sample-{milestone}.png", nrow=nrow)
                save_checkpoint(args.results_folder / f"model-{milestone}.pt", step, accelerator, diffusion, opt, ema)

    pbar.close()
    accelerator.print("classifier-free guidance training complete")


if __name__ == "__main__":
    main()
