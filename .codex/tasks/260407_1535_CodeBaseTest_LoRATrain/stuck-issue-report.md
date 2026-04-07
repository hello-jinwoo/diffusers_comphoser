# Stuck Issue Report

## Task Name
260407_1535_CodeBaseTest_LoRATrain

## Title
Multi-GPU post-training stall during validation / final inference

## Symptom
During multi-GPU LoRA training, the run could appear stuck after training reached the configured step limit. GPU utilization stayed near 100 percent and the process did not exit promptly.

## Reproduction Context
- command family: `accelerate launch --multi_gpu --num_processes=2 examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py ...`
- model: `black-forest-labs/FLUX.2-klein-4B`
- dataset: `./sample_data/flux2_klein_lora_smoke`
- observed during:
  - rank-0-only epoch validation
  - rank-0-only final inference after LoRA save

## Root Cause
The trainer executed validation and final inference only on `accelerator.is_main_process`, but there was no barrier bracketing that rank-0-only work.

That created an unsafe distributed flow:

1. rank 0 entered validation or final inference
2. other ranks continued toward the next distributed synchronization point
3. those ranks blocked in DDP / NCCL while rank 0 was still busy inside generation
4. from the outside, the job looked hung with GPUs still active

There was also an epoch-trigger detail that made validation run too early:

- previous logic: `epoch % args.validation_epochs == 0`
- consequence: validation ran at epoch `0` whenever `validation_prompt` was set

## Fix Applied
In `examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py`:

- changed epoch validation trigger to completed-epoch semantics:
  - `(epoch + 1) % args.validation_epochs == 0`
- added `accelerator.wait_for_everyone()` before rank-0-only epoch validation
- added `accelerator.wait_for_everyone()` after rank-0-only epoch validation
- added a final `accelerator.wait_for_everyone()` before `accelerator.end_training()`

## Validation Evidence
Validated after the fix with:

1. multi-GPU training plus final inference:
   - `--multi_gpu --num_processes=2`
   - `--max_train_steps=2`
   - no `--skip_final_inference`
   - result: LoRA save completed, final validation inference ran, process exited cleanly
2. multi-GPU training matching the simplified README behavior:
   - `--multi_gpu --num_processes=2`
   - end-to-end command form with validation prompt/image
   - result: process exited cleanly

## Operational Guidance
- Use the end-to-end README command when you want to validate the full training-plus-inference path.
- Use `--skip_final_inference` only when you explicitly want a faster training-only smoke run.
- If validation is enabled, high GPU usage after the last optimization step can still be legitimate while rank 0 is generating images, but the run should now terminate once that work finishes.
