"""Installable checkpoint-evaluation entrypoint for ComPhoser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from comphoser.evaluation import EVALUATION_MODES, CheckpointEvaluationConfig, evaluate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a primitive mode on a paired dataset split.")
    parser.add_argument(
        "--evaluation_mode",
        default="lora_qformer",
        choices=EVALUATION_MODES,
        help="Evaluation runtime. Use flux_only for the pretrained FLUX.2 Klein baseline without checkpoint files.",
    )
    parser.add_argument("--checkpoint_dir", type=Path, help="Checkpoint root containing LoRA and comphoser/ artifacts.")
    parser.add_argument("--dataset_root", required=True, type=Path, help="Contract dataset root or a train/val split root.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Report output directory under runs/reports/...")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Dataset split when --dataset_root is not a split root.")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional number of validation samples to evaluate.")
    parser.add_argument("--num_outputs_per_sample", type=int, default=1, help="Number of generated outputs per sample.")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Diffusion inference steps per output.")
    parser.add_argument("--seed", type=int, default=17, help="Base seed. Use a negative value to disable explicit seeding.")
    parser.add_argument("--resolution", type=int, default=None, help="Optional square inference resolution.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Pipeline and Q-Former dtype.",
    )
    parser.add_argument("--device", default="auto", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=None,
        help="Override the checkpoint backbone id, or the default pretrained model used by flux_only.",
    )
    parser.add_argument("--revision", default=None, help="Optional pretrained model revision.")
    parser.add_argument("--variant", default=None, help="Optional pretrained model variant.")
    parser.add_argument("--primitive_group", default=None, help="Primitive-group override for custom or ambiguous dataset roots.")
    parser.add_argument("--task_id", default=None, help="Task-id override for custom or ambiguous dataset roots.")
    parser.add_argument("--qformer_num_heads", type=int, default=16, help="Q-Former attention heads used to instantiate the controller.")
    parser.add_argument(
        "--explicit_token_masking",
        type=float,
        nargs=16,
        default=None,
        metavar="GATE",
        help="Optional 16-value runtime query-gate override for lora_qformer evaluation.",
    )
    parser.add_argument(
        "--disable_model_cpu_offload",
        action="store_true",
        help="Keep the pipeline on --device instead of using model CPU offload.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CheckpointEvaluationConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluation_mode == "lora_qformer" and args.checkpoint_dir is None:
        parser.error("--checkpoint_dir is required when --evaluation_mode=lora_qformer")
    if args.evaluation_mode == "flux_only" and args.checkpoint_dir is not None:
        parser.error("--checkpoint_dir is not used when --evaluation_mode=flux_only")
    seed = None if args.seed is not None and args.seed < 0 else args.seed
    return CheckpointEvaluationConfig(
        checkpoint_dir=args.checkpoint_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        evaluation_mode=args.evaluation_mode,
        split=args.split,
        sample_limit=args.sample_limit,
        num_outputs_per_sample=args.num_outputs_per_sample,
        num_inference_steps=args.num_inference_steps,
        seed=seed,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
        torch_dtype=args.torch_dtype,
        device=args.device,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        primitive_group=args.primitive_group,
        task_id=args.task_id,
        qformer_num_heads=args.qformer_num_heads,
        enable_model_cpu_offload=not args.disable_model_cpu_offload,
        explicit_token_masking=None
        if args.explicit_token_masking is None
        else tuple(float(value) for value in args.explicit_token_masking),
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = evaluate_checkpoint(parse_args(argv))
    print(result["metrics_path"])
    return 0


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
