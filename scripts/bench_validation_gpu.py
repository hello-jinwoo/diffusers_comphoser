"""Benchmark validation GPU utilization for ComPhoser.

Compares three configurations on a small validation sample using GPU:1:
  A. enable_model_cpu_offload=True  (current default during periodic validation)
  B. enable_model_cpu_offload=False (all components GPU-resident)
  C. configuration B + per-case caching of prompt embeds / cond latents / Q-Former
     gated bank across num_outputs_per_sample (proposed optimization)

Run:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
    python scripts/bench_validation_gpu.py \
      --checkpoint_dir <ckpt> \
      --dataset_root data/detail_sr__DIV2K \
      --sample_limit 3 \
      --num_outputs_per_sample 2

Writes a JSON report with per-phase timings and peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from comphoser.controls import resolve_control_selection
from comphoser.datasets import load_prepared_pilot_records
from comphoser.evaluation import (
    load_evaluation_pipeline,
    load_evaluation_qformer,
    read_checkpoint_metadata,
    resolve_evaluation_dataset,
)
from comphoser.image_utils import load_rgb_image
from comphoser.inference import (
    _controlled_validation_inference_context,
    build_controlled_validation_cases,
    prepare_pilot_inference_conditioning,
)
from comphoser.training import prepare_pilot_transformer_conditioning


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mem_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def time_block(label, results):
    class _Timer:
        def __enter__(self_inner):
            _sync()
            self_inner.t0 = time.perf_counter()
            return self_inner

        def __exit__(self_inner, *args):
            _sync()
            elapsed = time.perf_counter() - self_inner.t0
            results.setdefault(label, []).append(elapsed)

    return _Timer()


def run_naive(pipeline, qformer, cases, *, num_outputs, num_inference_steps, guidance_scale, seed):
    """Mirror current run_controlled_validation_case logic: re-encode prompt and cond per output."""
    timings = {}
    _reset_peak()
    images = []
    for case_index, case in enumerate(cases):
        for output_index in range(num_outputs):
            output_seed = seed + case_index * num_outputs + output_index
            generator = torch.Generator(device=pipeline._execution_device).manual_seed(output_seed)
            with time_block("per_output_total", timings):
                condition_image = load_rgb_image(case.condition_image_path)
                with _controlled_validation_inference_context(pipeline):
                    with time_block("encode_prompt_and_cond", timings):
                        conditioning = prepare_pilot_inference_conditioning(
                            pipeline,
                            case.prompt,
                            condition_image,
                            qformer=qformer,
                            primitive_groups=(case.primitive_family,) if case.primitive_family else (),
                            task_strengths=(float(case.task_strength),) if case.primitive_family else (),
                            generator=generator,
                            guidance_scale=guidance_scale,
                            height=None,
                            width=None,
                        )
                    with time_block("pipeline_call", timings):
                        image = pipeline(
                            image=condition_image,
                            prompt_embeds=conditioning.prompt_embeds,
                            negative_prompt_embeds=conditioning.negative_prompt_embeds,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            height=conditioning.height,
                            width=conditioning.width,
                        ).images[0]
                        images.append(image)
    timings["peak_mem_mb"] = _peak_mem_mb()
    return timings


def run_cached(pipeline, qformer, cases, *, num_outputs, num_inference_steps, guidance_scale, seed):
    """Compute prompt embeds / negative prompt embeds / cond image latents / Q-Former bank
    once per case and only re-run the denoising loop for each output seed."""
    timings = {}
    _reset_peak()
    images = []
    device = pipeline._execution_device
    for case_index, case in enumerate(cases):
        condition_image = load_rgb_image(case.condition_image_path)
        # Per-case work: encode prompt, negative prompt, cond image, Q-Former gating
        with _controlled_validation_inference_context(pipeline):
            with time_block("per_case_setup", timings):
                prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=case.prompt,
                    device=device,
                    num_images_per_prompt=1,
                )
                negative_prompt_embeds = None
                if guidance_scale > 1.0 and not getattr(pipeline.config, "is_distilled", False):
                    negative_prompt_embeds, _ = pipeline.encode_prompt(
                        prompt="",
                        device=device,
                        num_images_per_prompt=1,
                    )

                if qformer is not None and case.primitive_family:
                    # Encode condition image to latents once
                    from comphoser.inference import _prepare_condition_image_tensor

                    prepared, resolved_height, resolved_width = _prepare_condition_image_tensor(
                        pipeline, condition_image, height=None, width=None
                    )
                    cond_tokens, _ = pipeline.prepare_image_latents(
                        images=[prepared],
                        batch_size=1,
                        generator=None,
                        device=device,
                        dtype=pipeline.vae.dtype,
                    )
                    controls = resolve_control_selection(
                        primitive_groups=(case.primitive_family,),
                        task_strengths=(float(case.task_strength),),
                    )
                    conditioning = prepare_pilot_transformer_conditioning(
                        prompt_embeds,
                        text_ids,
                        cond_tokens,
                        qformer=qformer,
                        primitive_groups=controls.primitive_groups,
                        primitive_strengths=controls.primitive_strengths,
                    )
                    cached_prompt_embeds = conditioning.encoder_hidden_states
                else:
                    cached_prompt_embeds = prompt_embeds
                    resolved_height = resolved_width = None

            for output_index in range(num_outputs):
                output_seed = seed + case_index * num_outputs + output_index
                generator = torch.Generator(device=device).manual_seed(output_seed)
                with time_block("per_output_pipeline_call", timings):
                    image = pipeline(
                        image=condition_image,
                        prompt_embeds=cached_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        height=resolved_height,
                        width=resolved_width,
                    ).images[0]
                    images.append(image)
    timings["peak_mem_mb"] = _peak_mem_mb()
    return timings


def summarize(timings):
    summary = {}
    for key, value in timings.items():
        if isinstance(value, list):
            if value:
                summary[key] = {
                    "count": len(value),
                    "total_s": sum(value),
                    "mean_s": sum(value) / len(value),
                    "min_s": min(value),
                    "max_s": max(value),
                }
        else:
            summary[key] = value
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--sample_limit", type=int, default=3)
    parser.add_argument("--num_outputs_per_sample", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=("offload", "gpu_naive", "gpu_cached"),
        default=("offload", "gpu_naive", "gpu_cached"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metadata = read_checkpoint_metadata(args.checkpoint_dir)
    pretrained = metadata["backbone_id"]

    dataset = resolve_evaluation_dataset(args.dataset_root, split="val")
    records = load_prepared_pilot_records(dataset.dataset_root, split=dataset.split)
    cases = build_controlled_validation_cases(records, sample_limit=args.sample_limit)
    print(f"Using {len(cases)} validation cases from {dataset.dataset_id}")

    report = {
        "device": str(device),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_id": dataset.dataset_id,
        "num_cases": len(cases),
        "num_outputs_per_sample": args.num_outputs_per_sample,
        "num_inference_steps": args.num_inference_steps,
        "configs": {},
    }

    torch_dtype = torch.bfloat16

    for cfg in args.configs:
        print(f"\n=== Running config: {cfg} ===")
        enable_offload = cfg == "offload"
        with time_block("pipeline_load", report["configs"].setdefault(cfg, {})) as _:
            pipeline = load_evaluation_pipeline(
                pretrained_model_name_or_path=pretrained,
                checkpoint_dir=args.checkpoint_dir,
                load_lora_weights=True,
                torch_dtype=torch_dtype,
                device=device,
                enable_model_cpu_offload=enable_offload,
            )
            qformer = load_evaluation_qformer(
                args.checkpoint_dir,
                metadata=metadata,
                device=device,
                torch_dtype=torch_dtype,
            )
        config_report = report["configs"][cfg]
        # Warmup one call to amortize lazy init
        print("  warmup ...")
        warmup_case = cases[0]
        warmup_image = load_rgb_image(warmup_case.condition_image_path)
        with torch.inference_mode():
            with _controlled_validation_inference_context(pipeline):
                generator = torch.Generator(device=pipeline._execution_device).manual_seed(0)
                conditioning = prepare_pilot_inference_conditioning(
                    pipeline,
                    warmup_case.prompt,
                    warmup_image,
                    qformer=qformer,
                    primitive_groups=(warmup_case.primitive_family,) if warmup_case.primitive_family else (),
                    task_strengths=(float(warmup_case.task_strength),) if warmup_case.primitive_family else (),
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                )
                _ = pipeline(
                    image=warmup_image,
                    prompt_embeds=conditioning.prompt_embeds,
                    negative_prompt_embeds=conditioning.negative_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=conditioning.height,
                    width=conditioning.width,
                ).images[0]
        _reset_peak()

        if cfg == "gpu_cached":
            runner = run_cached
        else:
            runner = run_naive
        print(f"  running {len(cases)} cases x {args.num_outputs_per_sample} outputs ...")
        with torch.inference_mode():
            timings = runner(
                pipeline,
                qformer,
                cases,
                num_outputs=args.num_outputs_per_sample,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
        config_report.update(summarize(timings))
        print(f"  done: {json.dumps(config_report, indent=2, default=str)}")

        del pipeline, qformer
        torch.cuda.empty_cache()
        from diffusers.training_utils import free_memory

        free_memory()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nWrote report to {args.output_json}")


if __name__ == "__main__":
    main()
