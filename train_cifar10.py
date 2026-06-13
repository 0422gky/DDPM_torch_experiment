import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10 images.")

    parser.add_argument("--data-folder", type=Path, default=Path("data/cifar10/train"))
    parser.add_argument("--results-folder", type=Path, default=Path("results/cifar10"))
    parser.add_argument("--resume", type=str, default=None, help="Milestone to load, e.g. 50, best, or latest.")

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--flash-attn", action="store_true")

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=250)
    parser.add_argument("--objective", choices=("pred_noise", "pred_x0", "pred_v"), default="pred_v")
    parser.add_argument("--beta-schedule", choices=("linear", "cosine", "sigmoid"), default="sigmoid")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulate-every", type=int, default=1)
    parser.add_argument("--train-lr", type=float, default=8e-5)
    parser.add_argument("--train-num-steps", type=int, default=100000)
    parser.add_argument("--save-and-sample-every", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--mixed-precision-type", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--calculate-fid", action="store_true")
    parser.add_argument("--num-fid-samples", type=int, default=10000)
    parser.add_argument("--save-best-and-latest-only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

    args.results_folder.mkdir(parents=True, exist_ok=True)

    model = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=args.channels,
        flash_attn=args.flash_attn,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
    )

    trainer = Trainer(
        diffusion,
        str(args.data_folder),
        train_batch_size=args.batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        results_folder=str(args.results_folder),
        amp=not args.no_amp,
        mixed_precision_type=args.mixed_precision_type,
        ema_decay=args.ema_decay,
        max_grad_norm=args.max_grad_norm,
        calculate_fid=args.calculate_fid,
        num_fid_samples=args.num_fid_samples,
        save_best_and_latest_only=args.save_best_and_latest_only,
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
