#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageFilter, ImageOps


DEFAULT_DATASET = "huggan/smithsonian_butterflies_subset"
DEFAULT_SPLIT = "train[:4]"
DEFAULT_PROMPT = (
    "restore the butterfly photo from the blurred low-resolution reference while preserving natural wing patterns "
    "and color"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and build a tiny local img2img dataset for FLUX.2-Klein LoRA smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data/flux2_klein_lora_smoke"))
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_target_image(image: Image.Image, image_size: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.LANCZOS)


def prepare_conditioning_image(target_image: Image.Image, image_size: int) -> Image.Image:
    low_res_size = max(96, image_size // 3)
    return (
        target_image.resize((low_res_size, low_res_size), Image.Resampling.BICUBIC)
        .resize((image_size, image_size), Image.Resampling.BICUBIC)
        .filter(ImageFilter.GaussianBlur(radius=1.2))
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    train_dir = output_dir / "train"
    targets_dir = train_dir / "targets"
    conditioning_dir = train_dir / "conditioning"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(output_dir)

    targets_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    metadata_path = train_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, row in enumerate(dataset):
            target_image = prepare_target_image(row["image"], args.image_size)
            conditioning_image = prepare_conditioning_image(target_image, args.image_size)

            target_rel_path = Path("targets") / f"{index:03d}.png"
            conditioning_rel_path = Path("conditioning") / f"{index:03d}.png"

            target_image.save(train_dir / target_rel_path)
            conditioning_image.save(train_dir / conditioning_rel_path)

            metadata = {
                "file_name": str(target_rel_path),
                "conditioning_image": str(conditioning_rel_path),
                "instruction": args.prompt,
            }
            metadata_file.write(json.dumps(metadata) + "\n")

    print(f"Wrote {len(dataset)} paired samples to {output_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Validation image: {train_dir / Path('conditioning/000.png')}")


if __name__ == "__main__":
    main()
