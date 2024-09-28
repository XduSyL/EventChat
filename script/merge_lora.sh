python /data/SyL/Event_RGB/merge_lora/merge.py \
    --base_model_path /data/SyL/model/vicuna-7b-v1.5 \
    --lora_config /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-instruction/checkpoint-400/ \
    --lora_weights_path /data/SyL/Event_RGB/checkpoints/llava-v1.5-7b-instruction/checkpoint-400/lora_parameters.pth \
    --output_model_path /data/SyL/model/Event_RGB/vicuna-7b-v1.5-merge