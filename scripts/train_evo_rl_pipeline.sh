#!/usr/bin/env bash
# Run Evo-RL's value-training -> value-inference -> policy-training pipeline offline.
# All batch sizes are per GPU, matching the multi-GPU examples in Evo-RL's README.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT=""
DATASET_REPO_ID=""
PI05_PATH=""
PALIGEMMA_PATH=""
VALUE_VISION_PATH=""
VALUE_LANGUAGE_PATH=""
OUTPUT_ROOT="$CODE_DIR/outputs/evo-rl-runs"
JOB_NAME="evo-rl-$(date +%Y%m%d-%H%M%S)"
STAGES="value_train,value_infer,policy_train"
GPUS="0,1"
NUM_PROCESSES="2"

VALUE_STEPS="2"
VALUE_BATCH_SIZE="1"
VALUE_NUM_WORKERS="2"
INFER_BATCH_SIZE="2"
INFER_NUM_WORKERS="2"
POLICY_STEPS="2"
POLICY_BATCH_SIZE="1"
POLICY_NUM_WORKERS="2"
SAVE_FREQ="1"
LOG_FREQ="1"
ACP_TAG="value"
ACP_N_STEP="50"
ACP_POSITIVE_RATIO="0.3"
SWANLAB_MODE="cloud"
SWANLAB_PROJECT="evo-rl"
SWANLAB_WORKSPACE=""
COPY_DATASET="true"
VALUE_VIZ="false"
DRY_RUN="false"
VALUE_CHECKPOINT_PATH=""
ANNOTATED_DATASET_ROOT=""

usage() {
    cat <<'EOF'
Usage: train_evo_rl_pipeline.sh [options]

Runs the three Evo-RL stages in this order:
  value_train -> value_infer -> policy_train

Important:
  * --*-batch-size values are per GPU.
  * By default Value Inference writes ACP annotations into a copy under the run
    directory; the source dataset is never modified.
  * Hugging Face network access is disabled for every launched process.

Core paths:
  --code-dir PATH                 Evo-RL checkout
  --python PATH                   Python inside the evo-rl environment
  --dataset-root PATH             Source LeRobot dataset directory (required)
  --dataset-repo-id ID            Logical local dataset id (default: local/<dataset basename>)
  --pi05-path PATH                Local Pi05 pretrained-model directory (required for policy_train)
  --paligemma-path PATH           Local PaliGemma tokenizer directory (required for policy_train)
  --value-vision-path PATH        Local SigLIP backbone for Pi*0.6 (required for value_train)
  --value-language-path PATH      Local Gemma language backbone for Pi*0.6 (required for value_train)
  --output-root PATH              Parent directory for run outputs
  --job-name NAME                 Run directory name

Execution:
  --stages LIST                   Comma-separated subset: value_train,value_infer,policy_train
  --gpus IDS                      CUDA_VISIBLE_DEVICES list (default: 0,1)
  --num-processes N               accelerate process count (default: 2)
  --value-steps N                 Value training steps
  --value-batch-size N            Value per-GPU batch size
  --value-num-workers N
  --infer-batch-size N            Value inference per-GPU batch size
  --infer-num-workers N
  --policy-steps N                Policy training steps
  --policy-batch-size N           Policy per-GPU batch size
  --policy-num-workers N
  --save-freq N
  --log-freq N
  --value-checkpoint-path PATH    Existing value run/checkpoint when value_train is skipped
  --annotated-dataset-root PATH   Existing ACP-annotated data when value_infer is skipped
  --in-place-dataset              Permit Value Inference to mutate --dataset-root
  --value-viz                     Write Value Inference overlay videos to the run directory

ACP and SwanLab:
  --acp-tag NAME                  Suffix for complementary_info.{value,advantage,acp_indicator}_NAME
  --acp-n-step N
  --acp-positive-ratio FLOAT
  --swanlab-mode MODE             cloud (default), local, offline, or disabled
  --swanlab-project NAME
  --swanlab-workspace NAME
  --dry-run                        Print validated commands without launching stages
  -h, --help
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "Missing value for $1"
    printf '%s' "$2"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --code-dir) CODE_DIR="$(require_value "$@")"; shift 2 ;;
        --python) PYTHON="$(require_value "$@")"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$(require_value "$@")"; shift 2 ;;
        --dataset-repo-id) DATASET_REPO_ID="$(require_value "$@")"; shift 2 ;;
        --pi05-path|--policy-pretrained-path) PI05_PATH="$(require_value "$@")"; shift 2 ;;
        --paligemma-path) PALIGEMMA_PATH="$(require_value "$@")"; shift 2 ;;
        --value-vision-path) VALUE_VISION_PATH="$(require_value "$@")"; shift 2 ;;
        --value-language-path) VALUE_LANGUAGE_PATH="$(require_value "$@")"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$(require_value "$@")"; shift 2 ;;
        --job-name) JOB_NAME="$(require_value "$@")"; shift 2 ;;
        --stages) STAGES="$(require_value "$@")"; shift 2 ;;
        --gpus) GPUS="$(require_value "$@")"; shift 2 ;;
        --num-processes) NUM_PROCESSES="$(require_value "$@")"; shift 2 ;;
        --value-steps) VALUE_STEPS="$(require_value "$@")"; shift 2 ;;
        --value-batch-size) VALUE_BATCH_SIZE="$(require_value "$@")"; shift 2 ;;
        --value-num-workers) VALUE_NUM_WORKERS="$(require_value "$@")"; shift 2 ;;
        --infer-batch-size) INFER_BATCH_SIZE="$(require_value "$@")"; shift 2 ;;
        --infer-num-workers) INFER_NUM_WORKERS="$(require_value "$@")"; shift 2 ;;
        --policy-steps) POLICY_STEPS="$(require_value "$@")"; shift 2 ;;
        --policy-batch-size) POLICY_BATCH_SIZE="$(require_value "$@")"; shift 2 ;;
        --policy-num-workers) POLICY_NUM_WORKERS="$(require_value "$@")"; shift 2 ;;
        --save-freq) SAVE_FREQ="$(require_value "$@")"; shift 2 ;;
        --log-freq) LOG_FREQ="$(require_value "$@")"; shift 2 ;;
        --acp-tag) ACP_TAG="$(require_value "$@")"; shift 2 ;;
        --acp-n-step) ACP_N_STEP="$(require_value "$@")"; shift 2 ;;
        --acp-positive-ratio) ACP_POSITIVE_RATIO="$(require_value "$@")"; shift 2 ;;
        --swanlab-mode) SWANLAB_MODE="$(require_value "$@")"; shift 2 ;;
        --swanlab-project) SWANLAB_PROJECT="$(require_value "$@")"; shift 2 ;;
        --swanlab-workspace) SWANLAB_WORKSPACE="$(require_value "$@")"; shift 2 ;;
        --value-checkpoint-path) VALUE_CHECKPOINT_PATH="$(require_value "$@")"; shift 2 ;;
        --annotated-dataset-root) ANNOTATED_DATASET_ROOT="$(require_value "$@")"; shift 2 ;;
        --in-place-dataset) COPY_DATASET="false"; shift ;;
        --value-viz) VALUE_VIZ="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1. Use --help for usage." ;;
    esac
done

if [[ "$PYTHON" != */* ]]; then
    PYTHON="$(command -v "$PYTHON" || true)"
fi
[[ -n "$PYTHON" ]] || die "Python executable was not found; pass --python PATH or set PYTHON."

stage_enabled() {
    local wanted="$1" stage
    IFS=',' read -r -a stage_list <<< "$STAGES"
    for stage in "${stage_list[@]}"; do
        [[ "$stage" == "$wanted" ]] && return 0
    done
    return 1
}

require_dir() {
    [[ -d "$1" ]] || die "Directory not found: $1"
}

require_file() {
    [[ -f "$1" ]] || die "File not found: $1"
}

require_model_dir() {
    local label="$1" path="$2"
    require_dir "$path"
    require_file "$path/config.json"
    if ! compgen -G "$path/*.safetensors" >/dev/null && ! compgen -G "$path/pytorch_model*.bin" >/dev/null; then
        die "$label has no model weights under $path"
    fi
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

launch() {
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m accelerate.commands.launch \
        --multi_gpu \
        --num_processes="$NUM_PROCESSES" \
        --mixed_precision=bf16 \
        --no_python \
        "$PYTHON" "$@"
}

run_stage() {
    local stage="$1"
    shift
    local log_path="$RUN_DIR/${stage}.log"
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$stage" | tee -a "$RUN_DIR/pipeline.log"
    print_command "$@" | tee -a "$RUN_DIR/pipeline.log"
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    "$@" 2>&1 | tee "$log_path"
}

prepare_pi05_bundle() {
    local bundle="$RUN_DIR/pi05_pretrained_local"
    mkdir -p "$bundle"

    # Link large weights but keep the preprocessor JSON private to this run so its
    # tokenizer points to the supplied local PaliGemma directory.
    local item name
    for item in "$PI05_PATH"/*; do
        name="$(basename "$item")"
        [[ "$name" == "policy_preprocessor.json" ]] && continue
        ln -sfn "$item" "$bundle/$name"
    done
    cp "$PI05_PATH/policy_preprocessor.json" "$bundle/policy_preprocessor.json"
    "$PYTHON" - "$bundle/policy_preprocessor.json" "$PALIGEMMA_PATH" <<'PY'
import json
import sys

config_path, tokenizer_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as f:
    config = json.load(f)
found = False
for step in config.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        step.setdefault("config", {})["tokenizer_name"] = tokenizer_path
        found = True
if not found:
    raise RuntimeError(f"No tokenizer_processor step in {config_path}")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY
    printf '%s\n' "$bundle"
}

copy_dataset_tree() {
    local source="$1" destination="$2"
    mkdir -p "$destination"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "$source/" "$destination/"
    else
        # Cloud images often omit rsync. The run directory is new, so cp -a is
        # sufficient and keeps hard links, timestamps, and video files intact.
        cp -a "$source/." "$destination/"
    fi
}

[[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || die "--num-processes must be a positive integer"
[[ "$VALUE_STEPS" =~ ^[1-9][0-9]*$ ]] || die "--value-steps must be a positive integer"
[[ "$POLICY_STEPS" =~ ^[1-9][0-9]*$ ]] || die "--policy-steps must be a positive integer"
[[ "$SWANLAB_MODE" =~ ^(cloud|online|local|offline|disabled)$ ]] || die "Invalid --swanlab-mode: $SWANLAB_MODE"
require_dir "$CODE_DIR"
require_file "$CODE_DIR/src/lerobot/scripts/lerobot_value_train.py"
require_file "$PYTHON"
[[ -n "$DATASET_ROOT" ]] || die "--dataset-root is required"
require_dir "$DATASET_ROOT"
require_file "$DATASET_ROOT/meta/info.json"

if stage_enabled policy_train; then
    require_model_dir "Pi05 checkpoint" "$PI05_PATH"
    require_dir "$PALIGEMMA_PATH"
    require_file "$PALIGEMMA_PATH/tokenizer_config.json"
fi

if [[ "$SWANLAB_MODE" == "cloud" || "$SWANLAB_MODE" == "online" ]]; then
    [[ -n "${SWANLAB_API_KEY:-}" || -f "$HOME/.swanlab/.netrc" ]] || die \
        "SwanLab cloud mode needs SWANLAB_API_KEY or $HOME/.swanlab/.netrc"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LEROBOT_LOGGER=swanlab
export SWANLAB_MODE

gpu_count="$(tr -cd ',' <<< "$GPUS" | wc -c)"
gpu_count=$((gpu_count + 1))
[[ "$gpu_count" -eq "$NUM_PROCESSES" ]] || die \
    "--gpus ($GPUS) exposes $gpu_count GPU(s), but --num-processes is $NUM_PROCESSES"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -c '
import importlib.util
import torch
if importlib.util.find_spec("swanlab") is None:
    raise SystemExit("swanlab is not installed in the selected Python environment")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
print(f"PyTorch={torch.__version__}; visible CUDA devices={torch.cuda.device_count()}")
' || exit $?

if [[ -z "$DATASET_REPO_ID" ]]; then
    DATASET_REPO_ID="local/$(basename "$DATASET_ROOT")"
fi

RUN_DIR="$OUTPUT_ROOT/$JOB_NAME"
[[ ! -e "$RUN_DIR" ]] || die "Run directory already exists: $RUN_DIR (choose a new --job-name)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$0 $*" > "$RUN_DIR/command.txt"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'code_dir=%s\npython=%s\ndataset_root=%s\ndataset_repo_id=%s\ngpus=%s\nnum_processes=%s\n' \
        "$CODE_DIR" "$PYTHON" "$DATASET_ROOT" "$DATASET_REPO_ID" "$GPUS" "$NUM_PROCESSES"
    "$PYTHON" -c 'import accelerate, swanlab, torch, transformers; print(f"torch={torch.__version__}\naccelerate={accelerate.__version__}\ntransformers={transformers.__version__}\nswanlab={swanlab.__version__}")'
} > "$RUN_DIR/environment.txt"

if [[ -z "$ANNOTATED_DATASET_ROOT" ]]; then
    ANNOTATED_DATASET_ROOT="$RUN_DIR/dataset_acp"
fi
if [[ -z "$VALUE_CHECKPOINT_PATH" ]]; then
    VALUE_CHECKPOINT_PATH="$RUN_DIR/value_train"
fi

if stage_enabled value_train; then
    [[ -n "$VALUE_VISION_PATH" ]] || die \
        "--value-vision-path is required for offline Pi*0.6 value training"
    [[ -n "$VALUE_LANGUAGE_PATH" ]] || die \
        "--value-language-path is required for offline Pi*0.6 value training"
    require_model_dir "Pi*0.6 vision backbone" "$VALUE_VISION_PATH"
    require_model_dir "Pi*0.6 language backbone" "$VALUE_LANGUAGE_PATH"
fi

if stage_enabled value_train; then
    value_args=(
        -m lerobot.scripts.lerobot_value_train
        "--dataset.repo_id=$DATASET_REPO_ID"
        "--dataset.root=$DATASET_ROOT"
        --dataset.video_backend=torchcodec
        --value.type=pistar06
        "--value.vision_repo_id=$VALUE_VISION_PATH"
        "--value.language_repo_id=$VALUE_LANGUAGE_PATH"
        --value.dtype=bfloat16
        --value.push_to_hub=false
        --value.use_gradient_checkpointing=true
        "--value.scheduler_warmup_steps=0"
        "--value.scheduler_decay_steps=$VALUE_STEPS"
        "--output_dir=$RUN_DIR/value_train"
        "--job_name=$JOB_NAME.value"
        "--batch_size=$VALUE_BATCH_SIZE"
        "--steps=$VALUE_STEPS"
        "--save_freq=$SAVE_FREQ"
        "--log_freq=$LOG_FREQ"
        "--num_workers=$VALUE_NUM_WORKERS"
        --wandb.enable=true
        "--wandb.project=$SWANLAB_PROJECT"
        "--wandb.mode=$SWANLAB_MODE"
    )
    [[ -n "$SWANLAB_WORKSPACE" ]] && value_args+=("--wandb.entity=$SWANLAB_WORKSPACE")
    run_stage value_train launch "${value_args[@]}"
fi

if stage_enabled value_infer; then
    if [[ "$COPY_DATASET" == "true" ]]; then
        mkdir -p "$ANNOTATED_DATASET_ROOT"
        if [[ "$DRY_RUN" == "true" ]]; then
            printf '[dry-run] rsync source dataset to %s\n' "$ANNOTATED_DATASET_ROOT" | tee -a "$RUN_DIR/pipeline.log"
        else
            copy_dataset_tree "$DATASET_ROOT" "$ANNOTATED_DATASET_ROOT"
        fi
    else
        ANNOTATED_DATASET_ROOT="$DATASET_ROOT"
    fi

    infer_args=(
        -m lerobot.scripts.lerobot_value_infer
        "--dataset.repo_id=$DATASET_REPO_ID"
        "--dataset.root=$ANNOTATED_DATASET_ROOT"
        "--inference.checkpoint_path=$VALUE_CHECKPOINT_PATH"
        --inference.checkpoint_ref=last
        --runtime.device=cuda
        "--runtime.batch_size=$INFER_BATCH_SIZE"
        "--runtime.num_workers=$INFER_NUM_WORKERS"
        --acp.enable=true
        "--acp.n_step=$ACP_N_STEP"
        "--acp.positive_ratio=$ACP_POSITIVE_RATIO"
        "--acp.value_field=complementary_info.value_$ACP_TAG"
        "--acp.advantage_field=complementary_info.advantage_$ACP_TAG"
        "--acp.indicator_field=complementary_info.acp_indicator_$ACP_TAG"
        "--output_dir=$RUN_DIR/value_infer"
        "--job_name=$JOB_NAME.infer"
        "--viz.enable=$VALUE_VIZ"
        --viz.overwrite=true
    )
    run_stage value_infer launch "${infer_args[@]}"
fi

if stage_enabled policy_train; then
    require_dir "$ANNOTATED_DATASET_ROOT"
    PI05_LOCAL_BUNDLE="$(prepare_pi05_bundle)"
    policy_args=(
        -m lerobot.scripts.lerobot_train
        "--dataset.repo_id=$DATASET_REPO_ID"
        "--dataset.root=$ANNOTATED_DATASET_ROOT"
        --dataset.video_backend=torchcodec
        --policy.type=pi05
        "--policy.pretrained_path=$PI05_LOCAL_BUNDLE"
        --policy.device=cuda
        --policy.dtype=bfloat16
        --policy.gradient_checkpointing=true
        --policy.push_to_hub=false
        --policy.optimizer_lr=2.5e-05
        "--policy.scheduler_warmup_steps=0"
        "--policy.scheduler_decay_steps=$POLICY_STEPS"
        --policy.scheduler_decay_lr=2.5e-06
        --acp.enable=true
        "--acp.indicator_field=complementary_info.acp_indicator_$ACP_TAG"
        --acp.indicator_dropout_prob=0.3
        "--output_dir=$RUN_DIR/policy_train"
        "--job_name=$JOB_NAME.policy"
        "--batch_size=$POLICY_BATCH_SIZE"
        "--steps=$POLICY_STEPS"
        "--save_freq=$SAVE_FREQ"
        "--log_freq=$LOG_FREQ"
        "--num_workers=$POLICY_NUM_WORKERS"
        --wandb.enable=true
        "--wandb.project=$SWANLAB_PROJECT"
        "--wandb.mode=$SWANLAB_MODE"
    )
    [[ -n "$SWANLAB_WORKSPACE" ]] && policy_args+=("--wandb.entity=$SWANLAB_WORKSPACE")
    run_stage policy_train launch "${policy_args[@]}"
fi

printf '\nPipeline completed successfully: %s\n' "$RUN_DIR" | tee -a "$RUN_DIR/pipeline.log"
