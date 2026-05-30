"""Run ``build_preprocessed_dataset_from_raw`` on a contiguous slice of the
discovered samples.

Used to fan a single preprocessed build across multiple GPUs by giving each
process its own (offset, count) slice of the sample list. ``--if_exists skip``
keeps the two processes safe even if their slices overlap: a sample whose
output cache already exists is silently skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from comphoser import preprocessing as _pp  # noqa: E402


TORCH_DTYPE_CHOICES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _infer_dataset_name(dataset_root: Path) -> str:
    path = dataset_root.expanduser().resolve()
    if path.name in {"original", "raw", "preprocessed"}:
        path = path.parent
    if path.name in {"train", "val"}:
        return path.parent.name
    return path.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset_root", required=True, type=Path)
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--torch_dtype", choices=TORCH_DTYPE_CHOICES, default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--if_exists", choices=("error", "skip", "overwrite"), default="skip")
    p.add_argument("--sample_offset", type=int, default=0, help="0-based start index into the discovered sample list.")
    p.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Number of samples to process starting at offset (default: all remaining).",
    )
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--progress_every", type=int, default=200)
    p.add_argument(
        "--prompts_only",
        action="store_true",
        help="Skip image latent encoding; only rewrite the prompt latent cache .pt files. "
        "Used to refresh prompt encodings in place after rewriting the raw prompts.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch_dtype = TORCH_DTYPE_CHOICES[args.torch_dtype]
    if args.prompts_only:
        # load_flux_preprocessing_models still loads the VAE; prompts_only only skips the
        # image-latent encoding step and rewrites the prompt latent cache .pt files in place.
        tokenizer, text_encoder, _vae_unused = _pp.load_flux_preprocessing_models(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=args.device,
        )
        vae = None
    else:
        tokenizer, text_encoder, vae = _pp.load_flux_preprocessing_models(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=args.device,
        )
    dataset_name = _infer_dataset_name(args.dataset_root)
    samples = _pp.discover_raw_paired_samples(args.dataset_root, dataset_name=dataset_name)
    total = len(samples)
    start = max(0, args.sample_offset)
    end = total if args.sample_count is None else min(total, start + args.sample_count)
    slice_samples = samples[start:end]
    print(
        f"[preproc_slice] dataset={args.dataset_root} device={args.device} "
        f"total={total} slice={start}:{end} count={len(slice_samples)} "
        f"if_exists={args.if_exists}",
        flush=True,
    )

    import time

    written = 0
    skipped = 0
    t0 = time.time()
    for idx, sample in enumerate(slice_samples, 1):
        outs = sample.output_paths()
        # In prompts_only mode only the prompt latent path is considered "the cache".
        relevant_paths = (outs.prompt_latent_path,) if args.prompts_only else outs.all_paths()
        existing = [p for p in relevant_paths if p.exists()]
        if existing:
            if args.if_exists == "skip":
                if not args.prompts_only and len(existing) != len(relevant_paths):
                    raise FileExistsError(f"partial cache for sample {sample.sample_id}: {existing}")
                skipped += 1
                if args.progress_every and idx % args.progress_every == 0:
                    el = time.time() - t0
                    rate = idx / el if el > 0 else 0
                    print(
                        f"  [{idx}/{len(slice_samples)}] (skip-mode) written={written} "
                        f"skipped={skipped} rate={rate:.2f}/s",
                        flush=True,
                    )
                continue
            if args.if_exists == "error":
                raise FileExistsError(f"cache exists for sample {sample.sample_id}: {existing}")
            # overwrite: fall through.

        for d in (
            outs.input_image_latent_path.parent,
            outs.target_image_latent_path.parent,
            outs.prompt_latent_path.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)

        prompt = _pp.read_prompt_text(sample.prompt_path)
        pr_lat = _pp.encode_prompt_latent_with_flux_text_encoder(
            prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=args.device,
            max_sequence_length=args.max_sequence_length,
        )
        torch.save(pr_lat, outs.prompt_latent_path)

        if not args.prompts_only:
            in_tensor = _pp.load_and_preprocess_contract_image(sample.input_image_path, size=args.image_size)
            tg_tensor = _pp.load_and_preprocess_contract_image(sample.target_image_path, size=args.image_size)
            in_lat = _pp.encode_image_latent_with_flux_vae(in_tensor, vae=vae, device=args.device)
            tg_lat = _pp.encode_image_latent_with_flux_vae(tg_tensor, vae=vae, device=args.device)
            torch.save(in_lat, outs.input_image_latent_path)
            torch.save(tg_lat, outs.target_image_latent_path)

        written += 1

        if args.progress_every and idx % args.progress_every == 0:
            el = time.time() - t0
            rate = idx / el if el > 0 else 0
            print(
                f"  [{idx}/{len(slice_samples)}] written={written} skipped={skipped} "
                f"rate={rate:.2f}/s eta={(len(slice_samples) - idx) / max(rate, 1e-9):.0f}s",
                flush=True,
            )

    el = time.time() - t0
    print(
        f"[preproc_slice] done: dataset={args.dataset_root} slice=[{start}:{end}] "
        f"written={written} skipped={skipped} count={len(slice_samples)} "
        f"elapsed={el:.1f}s ({len(slice_samples) / max(el, 1e-9):.2f}/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
