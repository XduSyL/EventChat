import json

# 读取 thun_02_a_tracks.json 文件
with open('/data/SyL/model/thun_02_a_tracks.json', 'r') as f:
    tracks = json.load(f)

# 读取 thun_02_a_description.json 文件
with open('/data/SyL/model/thun_02_a_description.json', 'r') as f:
    descriptions = json.load(f)

# 遍历 descriptions 并添加 object_list
for i, description in enumerate(descriptions):
    description['object_list'] = tracks[i]['object_list']

# 将更新后的 descriptions 写回到 thun_02_a_description.json 文件
with open('thun_02_a_description.json', 'w') as f:
    json.dump(descriptions, f, indent=4)

print("object_list 已成功添加到 thun_02_a_description.json 中。")