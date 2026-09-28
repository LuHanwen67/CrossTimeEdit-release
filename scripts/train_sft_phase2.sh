#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-${ROOT}/external/DiffSynth-Studio}"
ACCELERATE="${ACCELERATE:-${ROOT}/.venv/bin/accelerate}"
MODEL_BASE="${MODEL_BASE:?Set MODEL_BASE to the FLUX.2-klein-base-4B directory}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the image root}"
SFT_MANIFEST="${SFT_MANIFEST:?Set SFT_MANIFEST to the SFT JSONL manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sft}"
TARGETS="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out"

for required in "${OUTPUT_DIR}/epoch-4.safetensors" "${OUTPUT_DIR}/checkpoints/epoch-4"; do
  [[ -e "${required}" ]] || { echo "Missing Phase-1 artifact: ${required}" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${DIFFSYNTH_DIR}"
"${ACCELERATE}" launch --mixed_precision bf16 --num_processes 4 --num_machines 1 \
  examples/flux2/model_training/train.py \
  --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${SFT_MANIFEST}" \
  --data_file_keys "image,edit_image" --extra_inputs "edit_image" \
  --height 512 --width 1024 --dataset_repeat 1 \
  --model_paths "[[\"${MODEL_BASE}/text_encoder/model-00001-of-00002.safetensors\",\"${MODEL_BASE}/text_encoder/model-00002-of-00002.safetensors\"],\"${MODEL_BASE}/transformer/diffusion_pytorch_model.safetensors\",\"${MODEL_BASE}/vae/diffusion_pytorch_model.safetensors\"]" \
  --tokenizer_path "${MODEL_BASE}/tokenizer" \
  --learning_rate 5e-5 --num_epochs 10 --gradient_accumulation_steps 4 \
  --lr_scheduler wsd --warmup_steps 300 --lr_decay_ratio 0.15 --min_lr_ratio 0.2 \
  --max_grad_norm 1.0 \
  --lora_base_model dit --lora_target_modules "${TARGETS}" --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." --output_path "${OUTPUT_DIR}" \
  --trainable_models dit --use_gradient_checkpointing --lr_reset
