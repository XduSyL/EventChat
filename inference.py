import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, CLIPVisionModel, LlamaForCausalLM, AutoConfig
import deepspeed
import argparse
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from dataset.IeTdataset import DSECDet, collate_fn, make_transforms
from torch.utils.data import DataLoader
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
from dataset.IeTdataset_transformers import make_supervised_data_module
from model.EventChatModel import EventChatModel
from dataset.IeTdataset_transformers import EventChatDataset, DataCollatorForEventChatDataset
from torch.utils.data import DataLoader
import time
import pdb
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import TextStreamer
from dataset.conversation import conv_templates, SeparatorStyle
from dataset.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER,DEFAULT_IMAGE_PATCH_TOKEN


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def generate_event_image(image, x, y, p):
    # 将 PIL 图像转换为 NumPy 数组
    image_np = np.array(image)
    # 获取图像的形状（高度和宽度）
    height, width = image_np.shape[:2]
    # 创建一个全白的事件图像，大小与 image 相同，三通道 RGB 图像
    event_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 绘制事件
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            event_image[y_, x_] = np.array([0, 0, 255])  # 蓝色事件
        else:
            event_image[y_, x_] = np.array([255, 0, 0])  # 红色事件

    return event_image


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='llava_v1')
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default='/data/SyL/Event_RGB/checkpoints/EventChat-v1.5-7b-instruction/checkpoint-400/mm_projector.bin')
    parser.add_argument("--event_frame", type=str, default='/data/SyL/LLaVA/custom_data/events/thun_02_a/000104.npy')

    args = parser.parse_args()

    config = AutoConfig.from_pretrained("/data/SyL/Event_RGB/checkpoints/EventChat-v1.5-7b-pretrain/checkpoint-500")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = EventChatModel.from_pretrained(args.model_path, torch_dtype=torch.float16, config=config)

    print("loading mm_projector pretrain")
    pretrained_weights = torch.load(args.pretrain_mm_mlp_adapter)
    pretrained_weights = {k.replace("model.visual_projector.", ""): v for k, v in pretrained_weights.items()}
    model.get_model().visual_projector.load_state_dict(pretrained_weights, strict=True)
    print("Pretrained weights loaded successfully into visual_projector.")

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_visual_tower()
    #vision_tower.load_model(device_map='auto')
    image_processor = vision_tower.image_processor
    context_len = 2048
                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    qs = args.query
    #model.config.mm_use_im_start_end = False
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(args.image_file)
    image_size = image.size

    event_npy = np.load(args.event_frame, allow_pickle=True)
    event_npy = np.array(event_npy).item()
    event_img = generate_event_image(image, event_npy['x'], event_npy['y'], event_npy['p'])

    event_image_pil = Image.fromarray(event_img)
    event_image_pil.save('event_frame_104.png')

    #image_processor = model.CLIPProcessor
    image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(device , dtype=torch.float16)

    event_tensor = image_processor(event_img, return_tensors='pt')['pixel_values']
    event_tensor = image_tensor.to(device , dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=event_tensor,
            image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            early_stopping=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


            

