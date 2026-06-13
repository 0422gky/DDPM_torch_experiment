import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class NoiseSchedule(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule="sigmoid"):
        super().__init__()

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_timesteps = int(timesteps)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ClassifierBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, downsample=False):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, dim_out)
        self.block = nn.Sequential(
            nn.GroupNorm(8, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.downsample = nn.Conv2d(dim_out, dim_out, 4, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block[0](x)
        h = self.block[1](h)
        h = self.block[2](h)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.block[3](h)
        h = self.block[4](h)
        h = self.block[5](h)
        return self.downsample(h + self.res_conv(x))


class NoisyCifar10Classifier(nn.Module):
    def __init__(self, dim=64, num_classes=10, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init = nn.Conv2d(3, dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            ClassifierBlock(dim, dim, time_dim, downsample=True),
            ClassifierBlock(dim, dim * 2, time_dim, downsample=True),
            ClassifierBlock(dim * 2, dim * 4, time_dim, downsample=True),
            ClassifierBlock(dim * 4, dim * 4, time_dim, downsample=False),
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(8, dim * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 4, num_classes),
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.init(x)
        for block in self.blocks:
            x = block(x, time_emb)
        return self.head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a noisy CIFAR-10 classifier for classifier guidance.")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10/raw"))
    parser.add_argument("--results-folder", type=Path, default=Path("results/cifar10_classifier"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=("linear", "cosine", "sigmoid"), default="sigmoid")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=2e-4)
    parser.add_argument("--train-num-steps", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize(args.image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    dataset = CIFAR10(root=str(args.data_root), train=True, download=True, transform=transform)
    dataloader = cycle(DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    ))

    schedule = NoiseSchedule(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)
    classifier = NoisyCifar10Classifier(dim=args.dim).to(device)
    opt = Adam(classifier.parameters(), lr=args.train_lr)

    step = 0
    if args.resume is not None:
        data = torch.load(args.resume, map_location=device, weights_only=True)
        classifier.load_state_dict(data["classifier"])
        opt.load_state_dict(data["opt"])
        step = int(data["step"])

    pbar = tqdm(initial=step, total=args.train_num_steps)
    while step < args.train_num_steps:
        images, labels = next(dataloader)
        images = normalize_to_neg_one_to_one(images.to(device))
        labels = labels.to(device)
        t = torch.randint(0, schedule.num_timesteps, (images.shape[0],), device=device).long()
        x_t = schedule.q_sample(images, t)

        logits = classifier(x_t, t)
        loss = F.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        acc = (logits.argmax(dim=1) == labels).float().mean()
        pbar.set_description(f"loss: {loss.item():.4f} acc: {acc.item():.3f}")
        pbar.update(1)

        if step % args.save_every == 0:
            milestone = step // args.save_every
            torch.save(
                {"step": step, "classifier": classifier.state_dict(), "opt": opt.state_dict()},
                args.results_folder / f"classifier-{milestone}.pt",
            )

    pbar.close()
    torch.save(
        {"step": step, "classifier": classifier.state_dict(), "opt": opt.state_dict()},
        args.results_folder / "classifier-latest.pt",
    )
    print("noisy classifier training complete")


if __name__ == "__main__":
    main()
