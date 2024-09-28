import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import matplotlib.pyplot as plt

# 加载模型和设备
model_id = "/data/SyL/model/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# 加载图像
image_path = "/data/SyL/dsec-det-master/event_img/000072.png"
image = Image.open(image_path)

# 设置检测文本
text = "a car. a house"

# 预处理输入
inputs = processor(images=image, text=text, return_tensors="pt").to(device)

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取检测结果
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print(results)

# 解析结果
detection_result = results[0]
boxes = detection_result["boxes"]
labels = detection_result["labels"]
scores = detection_result["scores"]

# 在图像上绘制检测框和标签
draw = ImageDraw.Draw(image)

# 加载一个更大的字体，设置大小为40（如果没有字体文件，也可以用默认字体）
try:
    font = ImageFont.truetype("arial.ttf", 300)  # 使用系统字体，可以更改路径
except IOError:
    font = ImageFont.load_default()  # 使用默认字体作为备选

for box, label, score in zip(boxes, labels, scores):
    box = box.cpu().numpy()  # 将box转为numpy数组
    
    # 绘制绿色矩形框，颜色为绿色，宽度为3
    draw.rectangle(box.tolist(), outline=(0, 128, 232), width=3)
    
    # 准备标签文本，包括类别和分数
    label_text = f"{label}: {score:.2f}"
    
    # 使用 `textbbox` 获取文本边界框
    text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # 计算文本宽度
    text_height = text_bbox[3] - text_bbox[1]  # 计算文本高度
    
    # 绘制黑色背景矩形框，用于放置文字（增加对比度）
    draw.rectangle(
        [box[0], box[1] - text_height, box[0] + text_width, box[1]], 
        fill="blue"
    )
    
    # 绘制白色文本
    draw.text((box[0], box[1] - text_height), label_text, fill="white", font=font)

# 保存绘制了bbox和标签的图像
output_image_path = "/data/SyL/Event_RGB/grounding_test/000072_with_bbox.png"
image.save(output_image_path)