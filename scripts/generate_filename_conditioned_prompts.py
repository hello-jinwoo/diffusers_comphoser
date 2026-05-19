"""Generate raw prompt text files from filename metadata for paired datasets."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Literal, Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _load_local_preprocessing_module():
    module_path = REPO_SRC / "comphoser" / "preprocessing.py"
    module_name = "_comphoser_preprocessing_filename_prompt_cli"
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

DatasetKind = Literal["realbokeh", "filmset"]

PAIRING_MODE_CHOICES = ("sorted", "order", "by_name")
IF_EXISTS_CHOICES = ("error", "skip", "overwrite")
_APERTURE_PATTERN = re.compile(r"_f([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)
_FILMSET_STYLE_NAMES = ("Cinema", "Velvia", "ClassNeg")

_REALBOKEH_BASE_PROMPTS = (
    "Apply realistic bokeh rendering while preserving the scene content and natural photographic detail.",
    "Edit the photo for depth-of-field control with natural bokeh rendering and a realistic lens look.",
    "Create a photographic depth edit with smooth bokeh rendering while keeping the subject and composition intact.",
)
_FILMSET_BASE_PROMPTS = (
    "Apply Stylization with the {style} Style and keep the photographic Filter Effect natural and coherent.",
    "Use {style} Style Stylization to create a consistent photographic Filter Effect while preserving scene detail.",
    "Create Stylization that matches the {style} Style with a clear photographic Filter Effect and natural tonality.",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw prompt text files aligned with raw sample ids using metadata encoded in filenames.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset split root such as data/depth_bokeh__RealBokeh/train or data/tone_style__FilmSet/val.",
    )
    parser.add_argument(
        "--dataset_kind",
        required=True,
        choices=("realbokeh", "filmset"),
        help="Prompt-generation rule set to apply.",
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


def _parse_aperture_value(image_path: Path) -> str:
    match = _APERTURE_PATTERN.search(image_path.stem)
    if match is None:
        raise ValueError(
            f"Could not infer aperture metadata from '{image_path.name}'. Expected a suffix like '_f2.8' or '_f22'."
        )
    return match.group(1)


def _parse_filmset_style(target_image_path: Path) -> str:
    stem = target_image_path.stem
    for style_name in _FILMSET_STYLE_NAMES:
        suffix = f"_{style_name}"
        if stem.endswith(suffix):
            return style_name
    raise ValueError(
        f"Could not infer FilmSet style metadata from '{target_image_path.name}'. "
        f"Expected one of: {', '.join(_FILMSET_STYLE_NAMES)}."
    )


def _build_realbokeh_prompt(input_image_path: Path, target_image_path: Path, sample_index: int) -> str:
    input_aperture = _parse_aperture_value(input_image_path)
    target_aperture = _parse_aperture_value(target_image_path)
    base_prompt = _REALBOKEH_BASE_PROMPTS[(sample_index - 1) % len(_REALBOKEH_BASE_PROMPTS)]
    aperture_prompt = (
        f"The input aperture look is f/{input_aperture}; adjust the bokeh rendering to match f/{target_aperture}."
    )
    return f"{base_prompt} {aperture_prompt}"


def _build_filmset_prompt(target_image_path: Path, sample_index: int) -> str:
    style_name = _parse_filmset_style(target_image_path)
    base_prompt = _FILMSET_BASE_PROMPTS[(sample_index - 1) % len(_FILMSET_BASE_PROMPTS)].format(style=style_name)
    style_prompt = f"The target Style is {style_name}, and the intended Filter Effect should clearly match that Style."
    return f"{base_prompt} {style_prompt}"


def _build_prompt_text(
    dataset_kind: DatasetKind,
    input_image_path: Path,
    target_image_path: Path,
    sample_index: int,
) -> str:
    if dataset_kind == "realbokeh":
        return _build_realbokeh_prompt(input_image_path, target_image_path, sample_index)
    if dataset_kind == "filmset":
        return _build_filmset_prompt(target_image_path, sample_index)
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


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
        prompt_text = _build_prompt_text(
            args.dataset_kind,
            sample.input_image_path,
            sample.target_image_path,
            sample_index,
        )
        output_paths.prompt_path.write_text(f"{prompt_text}\n", encoding="utf-8")
        written_sample_count += 1

    print(
        "Completed filename-conditioned prompt generation for "
        f"{dataset_name}: sample_count={len(samples)} "
        f"written={written_sample_count} skipped={skipped_sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
