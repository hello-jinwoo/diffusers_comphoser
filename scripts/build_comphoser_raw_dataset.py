"""Build contract-compliant raw image artifacts from original paired images."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _load_local_preprocessing_module():
    module_path = REPO_SRC / "comphoser" / "preprocessing.py"
    module_name = "_comphoser_preprocessing_raw_cli"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preprocessing module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PREPROCESSING = _load_local_preprocessing_module()
build_raw_images_from_original = _PREPROCESSING.build_raw_images_from_original


PAIRING_MODE_CHOICES = ("by_name", "sorted")
IF_EXISTS_CHOICES = ("error", "skip", "overwrite")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct raw image artifacts from original paired images without generating prompts.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset split root such as data/<dataname>/train or data/<dataname>/val.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Optional explicit dataset name prefix for output filenames.",
    )
    parser.add_argument(
        "--pairing_mode",
        choices=PAIRING_MODE_CHOICES,
        default="by_name",
        help="How to pair input and target images under original/images/.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Center-crop then resize each image to this square size.",
    )
    parser.add_argument(
        "--output_image_suffix",
        default=".jpg",
        help="Output image suffix for raw images.",
    )
    parser.add_argument(
        "--if_exists",
        choices=IF_EXISTS_CHOICES,
        default="error",
        help="How to handle existing raw image outputs.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional limit on the number of paired samples to process.",
    )
    parser.add_argument(
        "--sample_id_width",
        type=int,
        default=6,
        help="Zero-padded width for generated sample ids.",
    )
    return parser.parse_args(argv)


def _infer_dataset_name(dataset_root: str | Path, explicit_dataset_name: str | None) -> str | None:
    if explicit_dataset_name:
        return explicit_dataset_name

    path = Path(dataset_root).expanduser().resolve()
    if path.name in {"original", "raw", "preprocessed"}:
        path = path.parent
    if path.name in {"train", "val"}:
        return path.parent.name
    return path.name or None


def main(
    argv: Sequence[str] | None = None,
    *,
    build_raw_images_from_original_fn=build_raw_images_from_original,
) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root)
    dataset_name = _infer_dataset_name(dataset_root, args.dataset_name)
    result = build_raw_images_from_original_fn(
        dataset_root,
        dataset_name=dataset_name,
        pairing_mode=args.pairing_mode,
        image_size=args.image_size,
        output_image_suffix=args.output_image_suffix,
        if_exists=args.if_exists,
        sample_limit=args.sample_limit,
        sample_id_width=args.sample_id_width,
    )
    print(
        "Completed raw image dataset build for "
        f"{result.dataset_name}: sample_count={result.sample_count} "
        f"written={result.written_sample_count} skipped={result.skipped_sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
