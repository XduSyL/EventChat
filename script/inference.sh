#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
python /data/SyL/Event_RGB/inference.py \
    --pretrain_mm_mlp_adapter /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-pretrain/checkpoint-4000/mm_mlp_adapter.pth \
    --useLorafinetune False \
    --model_name_or_path /data/SyL/model/Event_RGB/vicuna-7b-v1.5-merge \
    --version plain \
    --image_path /data/SyL/LLaVA/data_process/merged_instruction.json \
    --vision_tower /data/SyL/model/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_mm_mlp_adapter /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-instruction \
