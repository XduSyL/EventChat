import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
import os

def merge_lora_with_base_model(
    base_model_path: str,
    lora_weights_path: str,
    output_model_path: str,
    lora_config_path: str
):
    # 加载基础模型
    print("加载基础模型...")
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载 LoRA 配置
    print("加载 LoRA 配置...")
    peft_config = PeftConfig.from_pretrained(lora_config_path)
    
    # 创建 PEFT 模型
    print("将 LoRA 配置应用到基础模型...")
    peft_model = PeftModel(base_model, peft_config)
    
    # 加载 LoRA 权重
    print("加载 LoRA 权重...")
    lora_state_dict = torch.load(lora_weights_path, map_location="cpu")
    lora_state_dict = {k.replace('base_model.model.', ''): v for k, v in lora_state_dict.items()}
    peft_model.load_state_dict(lora_state_dict, strict=False)
    
    # 融合 LoRA 权重到基础模型
    print("将 LoRA 权重与基础模型融合...")
    merged_model = peft_model.merge_and_unload()

    # 修复生成配置
    print("修复生成配置...")
    if hasattr(merged_model, "generation_config"):
        # 重置或删除无效参数
        generation_config = merged_model.generation_config
        if generation_config.do_sample is False:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None  # 根据需要也可以添加其他生成参数的清理

    # 保存融合后的模型
    print(f"保存融合后的模型到 {output_model_path}...")
    merged_model.generation_config.do_sample = True  # 或者
    merged_model.generation_config.temperature = None
    merged_model.generation_config.top_p = None
    merged_model.generation_config.top_k = None
    merged_model.save_pretrained(output_model_path)
    
    print("模型融合并保存成功。")

    print("保存 tokenizer 信息...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_model_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 LoRA 权重与基础模型融合。")
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型的路径。")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="LoRA 权重文件的路径（.pth 文件）。")
    parser.add_argument("--lora_config_path", type=str, required=True, help="LoRA 配置文件的路径。")
    parser.add_argument("--output_model_path", type=str, required=True, help="融合后模型的保存目录。")

    args = parser.parse_args()

    merge_lora_with_base_model(
        base_model_path=args.base_model_path,
        lora_weights_path=args.lora_weights_path,
        output_model_path=args.output_model_path,
        lora_config_path=args.lora_config_path
    )