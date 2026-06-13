import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from torchvision.datasets import CIFAR10


CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 and export it as a flat image folder for Trainer."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cifar10"),
        help="CIFAR-10 output root. Images are written under <root>/<split>.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to export for quick checks.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help="Optional mirror URL for cifar-10-python.tar.gz.",
    )
    return parser.parse_args()


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(url, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / CIFAR10_ARCHIVE

    if not archive_path.exists() or md5(archive_path) != CIFAR10_MD5:
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, archive_path)

    archive_md5 = md5(archive_path)
    if archive_md5 != CIFAR10_MD5:
        raise RuntimeError(
            f"bad md5 for {archive_path}: got {archive_md5}, expected {CIFAR10_MD5}"
        )

    extracted_dir = raw_dir / "cifar-10-batches-py"
    if not extracted_dir.exists():
        print(f"extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(raw_dir)


def main():
    args = parse_args()
    raw_dir = args.root / "raw"
    output_dir = args.root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_url is not None:
        download_and_extract(args.download_url, raw_dir)

    dataset = CIFAR10(
        root=str(raw_dir),
        train=args.split == "train",
        download=True,
    )

    class_names = dataset.classes
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    for index in range(total):
        image, label = dataset[index]
        class_name = class_names[label].replace(" ", "_")
        path = output_dir / f"{class_name}_{index:06d}.png"

        if path.exists() and not args.overwrite:
            continue

        image.save(path)

        if (index + 1) % 5000 == 0:
            print(f"exported {index + 1}/{total} images to {output_dir}")

    print(f"done: exported {total} CIFAR-10 {args.split} images to {output_dir}")


if __name__ == "__main__":
    main()
