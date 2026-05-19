"""Generate MSEC exposure-correction prompts aligned with raw dataset sample ids."""

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
    module_name = "_comphoser_preprocessing_msec_prompt_cli"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preprocessing module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PREPROCESSING = _load_local_preprocessing_module()
build_raw_sample_paths = _PREPROCESSING.build_raw_sample_paths
discover_original_paired_samples = _PREPROCESSING.discover_original_paired_samples

PAIRING_MODE_CHOICES = ("sorted", "order", "by_name")
IF_EXISTS_CHOICES = ("error", "skip", "overwrite")

_BASE_PROMPTS = (
    "Perform exposure correction to make a well-exposed image while preserving natural color and detail.",
    "Apply exposure correction to make a well-exposed image and keep the scene realistic, balanced, and natural.",
    "Use exposure correction to make a well-exposed image with natural tones and preserved image detail.",
)
_OFFSET_TAGS = ("P1.5", "P1", "N1.5", "N1", "0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MSEC raw prompt text files aligned with existing raw sample ids.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset split root such as data/exposure_ec__MSEC/train or data/exposure_ec__MSEC/val.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Optional explicit dataset name prefix for output filenames.",
    )
    parser.add_argument(
        "--pairing_mode",
        choices=PAIRING_MODE_CHOICES,
        default="order",
        help="How to reproduce the original sample ordering used for raw image sample ids.",
    )
    parser.add_argument(
        "--if_exists",
        choices=IF_EXISTS_CHOICES,
        default="error",
        help="How to handle existing raw prompt outputs.",
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


def _parse_exposure_offset_tag(input_image_path: Path) -> str:
    stem_upper = input_image_path.stem.upper()
    for tag in _OFFSET_TAGS:
        if stem_upper.endswith(f"_{tag}"):
            return tag
    raise ValueError(
        "Could not infer MSEC exposure direction from input filename "
        f"'{input_image_path.name}'. Expected suffixes like _P1, _P1.5, _N1, _N1.5, or _0."
    )


def _format_stop_count(value_text: str) -> str:
    return f"{value_text} stop" if value_text == "1" else f"{value_text} stops"


def _build_offset_prompt(tag: str) -> str:
    if tag.startswith("P"):
        stop_value = tag[1:]
        return (
            f"The input exposure offset is {tag}; lower the exposure by {_format_stop_count(stop_value)} "
            "to recover a balanced result."
        )
    if tag.startswith("N"):
        stop_value = tag[1:]
        return (
            f"The input exposure offset is {tag}; raise the exposure by {_format_stop_count(stop_value)} "
            "to recover a balanced result."
        )
    if tag == "0":
        return "The input exposure offset is 0; keep the exposure neutral and preserve the natural balance."
    raise ValueError(f"Unsupported MSEC exposure offset tag: {tag}")


def _build_prompt_text(input_image_path: Path, sample_index: int) -> str:
    offset_tag = _parse_exposure_offset_tag(input_image_path)
    prompt_parts = [_BASE_PROMPTS[(sample_index - 1) % len(_BASE_PROMPTS)]]
    prompt_parts.append(_build_offset_prompt(offset_tag))
    return " ".join(prompt_parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_id_width <= 0:
        raise ValueError("sample_id_width must be positive")

    dataset_root = Path(args.dataset_root)
    dataset_name = _infer_dataset_name(dataset_root, args.dataset_name)
    samples = discover_original_paired_samples(
        dataset_root,
        pairing_mode=args.pairing_mode,
        sample_limit=args.sample_limit,
        include_original_prompts=False,
    )

    written_sample_count = 0
    skipped_sample_count = 0
    for sample_index, sample in enumerate(samples, start=1):
        sample_id = f"{sample_index:0{args.sample_id_width}d}"
        output_paths = build_raw_sample_paths(
            dataset_root,
            dataset_name=dataset_name,
            sample_id=sample_id,
        )
        required_image_paths = (
            output_paths.input_image_path,
            output_paths.target_image_path,
        )
        missing_image_paths = tuple(path for path in required_image_paths if not path.is_file())
        if missing_image_paths:
            missing_text = ", ".join(str(path) for path in missing_image_paths)
            raise FileNotFoundError(
                "Raw images must exist before generating prompts. "
                f"Missing paths for sample '{sample_id}': {missing_text}"
            )

        if output_paths.prompt_path.exists():
            if args.if_exists == "overwrite":
                pass
            elif args.if_exists == "skip":
                skipped_sample_count += 1
                continue
            else:
                raise FileExistsError(f"Raw prompt already exists for sample '{sample_id}': {output_paths.prompt_path}")

        output_paths.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = _build_prompt_text(sample.input_image_path, sample_index)
        output_paths.prompt_path.write_text(f"{prompt_text}\n", encoding="utf-8")
        written_sample_count += 1

    print(
        "Completed MSEC prompt generation for "
        f"{dataset_name}: sample_count={len(samples)} "
        f"written={written_sample_count} skipped={skipped_sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
