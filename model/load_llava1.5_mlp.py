import torch

# 载入权重文件
weights_path = "/data/SyL/model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
new_weights_path = "/data/SyL/model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mlp_weights.pth"

# 加载权重
def inspect_weights(weights_path):
    try:
        # 使用torch.load加载权重
        checkpoint = torch.load(weights_path)
        
        # 如果权重是一个字典，通常会包含'state_dict'或其他键
        if isinstance(checkpoint, dict):
            print("权重文件包含以下键：")
            for key in checkpoint.keys():
                print(f"- {key}")
            
            # 如果存在'state_dict'，可以打印它的结构
            if 'state_dict' in checkpoint:
                print("\n'state_dict'中的参数：")
                for param_tensor in checkpoint['state_dict']:
                    print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())
            else:
                # 如果直接是参数
                print("\n权重文件中的参数：")
                for param_tensor in checkpoint:
                    print(param_tensor, "\t", checkpoint[param_tensor].size())
        else:
            # 如果不是字典，直接打印
            print("权重文件包含的参数：")
            for param_tensor in checkpoint:
                print(param_tensor, "\t", checkpoint[param_tensor].size())
    except Exception as e:
        print(f"读取权重文件时出错: {e}")

# 修改权重名称
def modify_weight_names(weights_path, new_weights_path):
    # 加载权重
    checkpoint = torch.load(weights_path)

    # 新的权重字典
    new_state_dict = {}

    # 原有的权重名称
    old_to_new = {
        "model.mm_projector.0.weight": "visual_projector.0.weight",
        "model.mm_projector.0.bias": "visual_projector.0.bias",
        "model.mm_projector.2.weight": "visual_projector.2.weight",
        "model.mm_projector.2.bias": "visual_projector.2.bias",
    }

    # 遍历权重字典，修改名称
    for old_name, param in checkpoint.items():
        if old_name in old_to_new:
            new_name = old_to_new[old_name]
            new_state_dict[new_name] = param
        else:
            new_state_dict[old_name] = param

    # 保存新的权重文件
    torch.save(new_state_dict, new_weights_path)
    print(f"新权重文件保存成功，路径：{new_weights_path}")


# 调用函数
#modify_weight_names(weights_path, new_weights_path)

# 调用函数查看权重文件
inspect_weights(new_weights_path)