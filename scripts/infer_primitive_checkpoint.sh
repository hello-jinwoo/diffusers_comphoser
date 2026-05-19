#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

EVALUATION_MODE="${EVALUATION_MODE:-lora_qformer}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DATASET_ROOT="${DATASET_ROOT:-data/depth_bokeh__RealBokeh/val}"
SPLIT="${SPLIT:-val}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
NUM_OUTPUTS_PER_SAMPLE="${NUM_OUTPUTS_PER_SAMPLE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
SEED="${SEED:-17}"
RESOLUTION="${RESOLUTION:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE="${DEVICE:-auto}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-}"
REVISION="${REVISION:-}"
VARIANT="${VARIANT:-}"
PRIMITIVE_GROUP="${PRIMITIVE_GROUP:-}"
TASK_ID="${TASK_ID:-}"
QFORMER_NUM_HEADS="${QFORMER_NUM_HEADS:-16}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-1}"

# GATING_TYPE controls runtime query gates in lora_qformer inference.
# - predicted_only: use Q-Former predicted gates.
# - all_zero/all_one: force every one of the 16 query gates to 0 or 1.
# - detail/tone/exposure/depth: force only that primitive family's 4 slots on.
# - custom: use EXPLICIT_TOKEN_MASKING with exactly 16 float values.
GATING_TYPE="${GATING_TYPE:-predicted_only}"
EXPLICIT_TOKEN_MASKING="${EXPLICIT_TOKEN_MASKING:-}"

dataset_name="$(basename "${DATASET_ROOT}")"
if [[ "${dataset_name}" == "train" || "${dataset_name}" == "val" ]]; then
  dataset_name="$(basename "$(dirname "${DATASET_ROOT}")")"
fi
RUN_TAG="${RUN_TAG:-${dataset_name}_${EVALUATION_MODE}_${GATING_TYPE}}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/reports/primitive/${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${EVALUATION_MODE}" == "lora_qformer" && -z "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR is required when EVALUATION_MODE=lora_qformer." >&2
  exit 1
fi
if [[ "${EVALUATION_MODE}" == "flux_only" && -n "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR must be empty when EVALUATION_MODE=flux_only." >&2
  exit 1
fi

EXPLICIT_GATE_ARGS=()
case "${GATING_TYPE}" in
  predicted_only)
    ;;
  all_zero)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
    ;;
  all_one)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
    ;;
  detail)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0)
    ;;
  tone)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0)
    ;;
  exposure)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0)
    ;;
  depth)
    EXPLICIT_GATE_ARGS=(--explicit_token_masking 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1)
    ;;
  custom)
    if [[ -z "${EXPLICIT_TOKEN_MASKING}" ]]; then
      echo "EXPLICIT_TOKEN_MASKING is required when GATING_TYPE=custom." >&2
      exit 1
    fi
    read -r -a CUSTOM_GATE_ARRAY <<< "${EXPLICIT_TOKEN_MASKING}"
    if [[ "${#CUSTOM_GATE_ARRAY[@]}" -ne 16 ]]; then
      echo "EXPLICIT_TOKEN_MASKING must contain exactly 16 values; received ${#CUSTOM_GATE_ARRAY[@]}." >&2
      exit 1
    fi
    EXPLICIT_GATE_ARGS=(--explicit_token_masking "${CUSTOM_GATE_ARRAY[@]}")
    ;;
  *)
    echo "Unsupported GATING_TYPE='${GATING_TYPE}'. Use predicted_only, all_zero, all_one, detail, tone, exposure, depth, or custom." >&2
    exit 1
    ;;
esac

if [[ "${EVALUATION_MODE}" == "flux_only" && "${#EXPLICIT_GATE_ARGS[@]}" -gt 0 ]]; then
  echo "GATING_TYPE=${GATING_TYPE} requires lora_qformer; flux_only has no Q-Former gates." >&2
  exit 1
fi

EVALUATOR=(
  -m comphoser.cli.evaluate_checkpoint
  --evaluation_mode "${EVALUATION_MODE}"
  --dataset_root "${DATASET_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --num_outputs_per_sample "${NUM_OUTPUTS_PER_SAMPLE}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --torch_dtype "${TORCH_DTYPE}"
  --device "${DEVICE}"
  --qformer_num_heads "${QFORMER_NUM_HEADS}"
)

if [[ -n "${CHECKPOINT_DIR}" ]]; then
  EVALUATOR+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  EVALUATOR+=(--sample_limit "${SAMPLE_LIMIT}")
fi
if [[ -n "${RESOLUTION}" ]]; then
  EVALUATOR+=(--resolution "${RESOLUTION}")
fi
if [[ -n "${PRETRAINED_MODEL_NAME_OR_PATH}" ]]; then
  EVALUATOR+=(--pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${REVISION}" ]]; then
  EVALUATOR+=(--revision "${REVISION}")
fi
if [[ -n "${VARIANT}" ]]; then
  EVALUATOR+=(--variant "${VARIANT}")
fi
if [[ -n "${PRIMITIVE_GROUP}" ]]; then
  EVALUATOR+=(--primitive_group "${PRIMITIVE_GROUP}")
fi
if [[ -n "${TASK_ID}" ]]; then
  EVALUATOR+=(--task_id "${TASK_ID}")
fi
if [[ "${ENABLE_MODEL_CPU_OFFLOAD}" != "1" ]]; then
  EVALUATOR+=(--disable_model_cpu_offload)
fi
if [[ "${#EXPLICIT_GATE_ARGS[@]}" -gt 0 ]]; then
  EVALUATOR+=("${EXPLICIT_GATE_ARGS[@]}")
fi

echo "Launching ComPhoser checkpoint inference/evaluation"
echo "  mode: ${EVALUATION_MODE}"
echo "  checkpoint_dir: ${CHECKPOINT_DIR:-<none>}"
echo "  dataset_root: ${DATASET_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  gating_type: ${GATING_TYPE}"
echo "  metrics: PSNR, SSIM, Delta E 2000"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'python'
  printf ' %q' "${EVALUATOR[@]}"
  printf '\n'
  exit 0
fi

python "${EVALUATOR[@]}"
