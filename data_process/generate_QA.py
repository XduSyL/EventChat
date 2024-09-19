import requests
from tqdm import tqdm
import json

def local_vllm(prompt):
    modelname = '/data/SyL/model/Qwen2-72B-Instruct-GPTQ-Int4'
    headers = {'Content-Type': 'application/json'}
    Qwen_data = {
        "model": "modeltype",
        "temperature": 0.8,
        "top_p": 0.8,
        "messages": [
            # {"role": "system", "content": "请输入您的instruction "},
            {"role": "user", "content": "请输出您的query"}
        ],
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }
    Qwen_data["model"] = modelname
    Qwen_data["messages"][0]["content"] = prompt
    response = requests.post('http://192.168.5.112:8111/v1/chat/completions', json=Qwen_data, headers=headers)
    res = response.text
    res = json.loads(res)
    res = res["choices"][0]["message"]["content"]
    print(res)
    return res

QA_prompt = """You are an AI visual assistant. What you see is an overall description of the content in the image, a description of the objects in the image, and the categories of the objects displayed in the image. then, you need to design queries that can be completed with objects that are shown in the image.

You should generate query-answeer pairs that are goal-orienated, where the user inputs something he/she wishes to achieve, and you are responsible to find the objects in the image that helps he/she to do so.  Also, the queries for an image should be diverse, spanning across all types of queriesmentioned above. 

The queries and their corresponding answers should be strongly correlated, with a focus on quality over quantity for each pair and keep your queries and answers as concise as possible

Note that you should design attribute-related queries (such as color or shape), only when you are certain about it. Do not generate such queries if the captions provided to you does not contain such information.

When answer each query, you must (1) describe all the object (you may refer to the complete object list), (2) based on common sense, use correct object(s) to answer the question, (3) list out target objects in the following manner: "Therefore the answer is: ⟨TARGET_OBJETCTS⟩".

You should answer the query based on the your understanding visual feature.
{image_description_str}
{object_desciption_str}
{object_list_str}
Note: 
1. The objects in ⟨TARGET_OBJETCTS⟩ must be visible in the image and can be used to solve the query in ⟨QUERY⟩. They also need to exist in Object list.
2. You must response any queries or answer in the following way: Query: ⟨QUERY⟩ Answer: ⟨ANSWER⟩ Therefore the answer is: ⟨TARGET_OBJETCTS⟩
3. The query-answer pairs generated should avoid duplication as much as possible.
Let's step by step

For Example:
Find the person on the road in the picture.
There is a person riding a bicycle on the road and a person walking on the road with a backpack on his back. Therefore the answer is: [rider, pedestrian]

Find all the transportation in the image.
In the image, bicycle and cycling can be used as means of transportation. Therefore the answer is: [car, bicycle]

Find a low-carbon travel method that can take into account both exercise and low-carbon travel.
In the image, there is a person riding a bicycle. This is a low-carbon lifestyle. Therefore the answer is: [rider]

Find all the objects present in the image.
This image contains pedestrians, bicycles, riders, and cars. Therefore the answer is: [car, bicycle, rider, pedestrian]

Special note: You must ensure that the objects described in the query and answer you give exist in the object_list provided to you, otherwise you will be killed!
"""

description_json_path = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/thun_01_a_description.json"
output_dir = "/data/SyL/Event_RGB/dataset/dsec-dataset/val/thun_01_a/"
sences = "thun_01_a_"
output_json_path = output_dir + sences + "QA_Dataset.json"

QA_Datasets_list = []
with open(description_json_path, "r") as f:
    description_json = json.load(f)

# Initialize the JSON file with an empty array
with open(output_json_path, 'w') as f:
    json.dump([], f)

for item in tqdm(description_json):
    image_desciption = item["image_description"]
    object_desciption_list = item["object_description"]
    object_desciption = ""
    for od in object_desciption_list:
        object_desciption += od + "\n"

    Image_description = "Image description:\n" + image_desciption + "\n"
    Object_description = "Object description:\n" + object_desciption + "\n"
    object_list = "Object list:\n" + item["object_list"] + "\n"
    prompt = QA_prompt
    prompt = QA_prompt.format(
    image_description_str=Image_description,
    object_desciption_str=Object_description,
    object_list_str=object_list
    )

    # print(prompt)
    response = local_vllm(prompt)

    QA_Datasets = {
        "sences": item["scenes"],
        "image_id": item["image_id"],
        "object_list": item["object_list"],
        "QA": response
    }

    # Read the existing JSON file
    with open(output_json_path, 'r') as f:
        existing_data = json.load(f)

    # Append the new QA_Datasets to the existing data
    existing_data.append(QA_Datasets)

    # Write the updated data back to the JSON file
    with open(output_json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


