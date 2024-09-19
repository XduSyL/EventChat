import numpy as np
import cv2
from tqdm import tqdm
import json
def number_to_word(number):
    class_map = {
        0: 'pedestrian',
        1: 'rider',
        2: 'car',
        3: 'bus',
        4: 'truck',
        5: 'bicycle',
        6: 'motorcycle',
        7: 'train'
    }

    if number in class_map:
        return class_map[number]
    else:
        return "Unknown"

def convert_numbers_to_unique_classes_string(numbers):
    # 定义类别映射
    class_map = {
        0: 'pedestrian',
        1: 'rider',
        2: 'car',
        3: 'bus',
        4: 'truck',
        5: 'bicycle',
        6: 'motorcycle',
        7: 'train'
    }

    # 使用列表和一个辅助集合来保持顺序并去重
    seen = set()
    unique_classes = []
    for number in numbers:
        class_name = class_map.get(number, 'unknown')
        if class_name not in seen:
            seen.add(class_name)
            unique_classes.append(class_name)

    # 将列表转换为逗号分隔的字符串
    return ', '.join(unique_classes)

show_bbox = False
path_dir = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/"
scenes = "thun_01_a/"
path = path_dir + scenes
# 读取标签
label_path = path + "object_detections/left/tracks.npy"
timestamps_path = path + "images/timestamps.txt"
image_path = path + "images/left/distorted/"
output_path = '/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/'
with open(timestamps_path, "r") as f:
    captions = f.readlines()

label = np.load(label_path)
t_track = np.array([tup[0] for tup in label])
t_image = np.array(list(map(int, [line.strip() for line in captions])))
x, counts = np.unique(t_track, return_counts=True)
i, j = (x.reshape((-1, 1)) == t_image.reshape((1, -1))).nonzero()
deltas = np.zeros_like(t_image)
deltas[j] = counts[i]
idx = np.concatenate([np.array([0]), deltas]).cumsum()
img_idx_to_track_idx = np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")
object_list = []

for i in tqdm(range(len(captions) - 1), desc="处理图片", unit="图片"):
    img_name = image_path + str(i).zfill(6) + '.png'
    img = cv2.imread(img_name)
    idx0, idx1 = img_idx_to_track_idx[i + 1]
    tracks = label[idx0:idx1]
    bbox_list = []
    class_list = []
    class_and_bbox_list = []

    for item in tracks:
        class_id = item[5]
        class_list.append(class_id)
        bbox = [int(item[1]), int(item[2]), int(item[3]), int(item[4])]  # 确保坐标是整数类型
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 使用蓝色绘制边界框
        # cv2.imshow('Image with BBox', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        class_and_bbox = "class : " + str(number_to_word(class_id)) + ", " + f"<box>({x1}, {y1}), ({x2}, {y2})</box>"
        class_and_bbox_list.append(class_and_bbox)
        #print(text)
        bbox_list.append(bbox)

    class_object = convert_numbers_to_unique_classes_string(class_list)

    object = {
        "scenes": scenes[:-1],
        "image_id": "",
        "object_list": "",
        "class_and_bbox": ""
    }

    image_id = str(i).zfill(6) + '.png'
    object["image_id"] = image_id
    object["class_and_bbox"] = class_and_bbox_list
    object["object_list"] = class_object
    object_list.append(object)

    if show_bbox == True:
        if i == 0 or i == (len(captions) - 1) // 2 or i == (len(captions) - 1) - 1:
            # 绘制bbox
            for bbox, class_id in zip(bbox_list, class_list):
                x, y, w, h = bbox
                # 计算矩形框的左上角和右下角坐标
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 使用蓝色绘制边界框
                # 绘制类名
                class_text = str(number_to_word(class_id))
                cv2.putText(img, class_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 使用红色绘制类名文本
            # 显示图像
            cv2.imshow('Image with BBox', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# 将 object_list 写入 JSON 文件
object_list_name = output_path + scenes[:-1] + "_tracks.json"
with open(object_list_name, 'w') as json_file:
    json.dump(object_list, json_file, indent=4)
