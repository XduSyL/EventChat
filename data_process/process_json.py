import json

# 读取原始 JSON 文件
input_file = '/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_02_a/QADataset.json'
output_file = '/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_02_a/labels.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# 新的 JSON 数组
new_data = []

# 遍历原始 JSON 数组中的每一个元素
for item in data:
    # 遍历每个元素中的 QA 列表
    for qa in item["QA"]:
        # 创建新的 JSON 对象，包含 Query 和 Answer 字段
        new_item = {
            "sences": item["sences"],
            "image_id": item["image_id"],
            "object_list": item["object_list"],
            "Query": qa["Query"],
            "Answer": qa["Answer"]
        }
        # 将新的 JSON 对象添加到新的 JSON 数组中
        new_data.append(new_item)

# 将新的 JSON 数组保存为文件
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=4)

print("新的 JSON 文件已生成：new_data.json")