import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from train_cifar10_noisy_classifier import NoisyCifar10Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Sample CIFAR-10 with classifier guidance.")
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_classifier_guided/sample.png"))

    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.0)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--classifier-dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=None)
    parser.add_argument("--objective", choices=("pred_noise", "pred_x0", "pred_v"), default="pred_v")
    parser.add_argument("--beta-schedule", choices=("linear", "cosine", "sigmoid"), default="sigmoid")
    return parser.parse_args()


def build_diffusion(args):
    from denoising_diffusion_pytorch import GaussianDiffusion, Unet

    model = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=3,
        flash_attn=False,
    )
    return GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
    )


def classifier_gradient(classifier, x, t, classes, guidance_scale):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[torch.arange(x.shape[0], device=x.device), classes].sum()
        grad = torch.autograd.grad(selected, x_in)[0]
    return grad * guidance_scale


def sample(diffusion, classifier, classes, guidance_scale):
    device = diffusion.betas.device
    batch = classes.shape[0]
    shape = (batch, diffusion.channels, diffusion.image_size[0], diffusion.image_size[1])
    img = torch.randn(shape, device=device)
    x_start = None

    for step in tqdm(reversed(range(diffusion.num_timesteps)), total=diffusion.num_timesteps, desc="classifier guided sampling"):
        t = torch.full((batch,), step, device=device, dtype=torch.long)
        self_cond = x_start if diffusion.self_condition else None

        with torch.no_grad():
            model_mean, variance, model_log_variance, x_start = diffusion.p_mean_variance(
                x=img,
                t=t,
                x_self_cond=self_cond,
                clip_denoised=True,
            )

        grad = classifier_gradient(classifier, model_mean, t, classes, guidance_scale)
        guided_mean = model_mean + variance * grad
        noise = torch.randn_like(img) if step > 0 else 0.0
        img = guided_mean + (0.5 * model_log_variance).exp() * noise

    return diffusion.unnormalize(img)


def main():
    args = parse_args()

    from ema_pytorch import EMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = build_diffusion(args).to(device)
    diffusion_data = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=True)
    ema = EMA(diffusion, beta=0.995, update_every=10)
    ema.load_state_dict(diffusion_data["ema"])
    diffusion = ema.ema_model.to(device).eval()

    classifier = NoisyCifar10Classifier(dim=args.classifier_dim).to(device)
    classifier_data = torch.load(args.classifier_checkpoint, map_location=device, weights_only=True)
    classifier.load_state_dict(classifier_data["classifier"])
    classifier.eval()

    classes = torch.tensor(args.classes, device=device, dtype=torch.long)
    classes = classes.repeat_interleave(args.samples_per_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = sample(diffusion, classifier, classes, args.guidance_scale)
    save_image(samples, args.output, nrow=args.samples_per_class)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
