import os
from lmdeploy.vl import load_image
from lmdeploy import pipeline, TurbomindEngineConfig
import json
from tqdm import tqdm

def deploy_internVL(pipe, prompt, image_path):

    image = load_image(image_path)

    response = pipe((prompt, image))
    print(response.text)
    return response.text

#Set CUDA_VISIBLE_DEVICES to limit to specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

tracks_path = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/thun_01_a_tracks.json"
image_dir = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/images/left/distorted/"
model_path = '/data/SyL/model/InternVL2-8B'
output_path = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/thun_01_a_description.json"

# Initialize the pipeline
pipe = pipeline(
    model_path=model_path,
    cache_max_entry_count=0.6,
    top_p=0.8,
    temperature = 0.8,
    backend_config=TurbomindEngineConfig(tp=4)
)

object_description_prompt = "The bounding box of an object in the image has been provided, please provide a brief description of the object.\n\nobject:\n\nNote! Just describe it directly, don't mention the bounding box"
image_description_prompt = """You are an AI visual assistant that can analyze a single image. You will provide a detailed description of the objects in this image, and you need to focus on the object information in the image

1. Provide a detailed description of the main elements in the image, such as pedestrian, rider, car, bus, truck, bicycle, motorcycle, train, etc., as well as their appearance, movements, and positions, while paying attention to color, texture, and other key details.
2. Please do not describe objects that do not appear in the picture
3. You are responsible for the following: first, you need to describe the image contents with  necessary but not redundant details.

You must organize your response into only one paragraph.
"""

with open(tracks_path, 'r') as f:
    tracks = json.load(f)

# Initialize the JSON file with an empty array
with open(output_path, 'w') as f:
    json.dump([], f)

for item in tqdm(tracks, desc="Processing images"):
    image_path = image_dir + item["image_id"]
    class_and_bbox_list = item["class_and_bbox"]
    image_description = deploy_internVL(pipe, image_description_prompt, image_path)
    object_description_list = []
    for class_and_bbox in class_and_bbox_list:
        prompt = object_description_prompt + class_and_bbox
        object_description = deploy_internVL(pipe, prompt, image_path)
        object_description_list.append(object_description)
    description_dataset = {
        "scenes": item["scenes"],
        "image_id": item["image_id"],
        "object_list": item["object_list"],
        "image_description": image_description,
        "object_description": object_description_list
    }

    # Append the new description_dataset to the JSON file
    with open(output_path, 'r') as f:
        description_datasets = json.load(f)
    description_datasets.append(description_dataset)
    with open(output_path, 'w') as f:
        json.dump(description_datasets, f, indent=4)