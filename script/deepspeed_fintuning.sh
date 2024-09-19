# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
deepspeed --include=localhost:4,5,6,7 /data/SyL/Event_RGB/deepspeed_train.py \
    --deepspeed /data/SyL/LLaVA/scripts/zero2.json \
    --pretrain_mm_mlp_adapter /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-pretrain/checkpoint-4000/mm_mlp_adapter.pth \
    --useLorafinetune True \
    --model_name_or_path /data/SyL/model/vicuna-7b-v1.5 \
    --version plain \
    --data_path  /data/SyL/LLaVA/data_process/merged_instruction.json \
    --image_folder /data/SyL/LLaVA/custom_data/images \
    --vision_tower /data/SyL/model/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-instruction \
    --output_mm_mlp_adapter /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-instruction \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

#    --event_folder /data/SyL/LLaVA/custom_data/events \