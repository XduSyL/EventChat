import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def load_and_generate_from_model(model_path: str, prompt: str):
    # 加载融合后的模型
    print(f"加载模型自 {model_path}...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载对应的 tokenizer
    print("加载 tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # 准备输入数据
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成文本
    print("生成文本...")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 输出生成的文本
    print(f"生成的文本: {generated_text}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="载入融合后的模型并生成文本。")
    parser.add_argument("--model_path", type=str, required=True, help="融合后模型的路径。")
    parser.add_argument("--prompt", type=str, default="你好，世界！", help="生成文本的提示语。")

    args = parser.parse_args()

    load_and_generate_from_model(
        model_path=args.model_path,
        prompt=args.prompt
    )