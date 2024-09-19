import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, CLIPVisionModel, LlamaForCausalLM
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


local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    output_mm_mlp_adapter: Optional[str] = field(default=None)
    useLorafinetune : bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    event_folder: Optional[str] = field(default=None)
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    # 可根据需要计算额外的度量指标
    return {"accuracy": (predictions == labels).mean().item()}


if __name__ == '__main__':
    
    #huggingface的库函数，将命令行参数解析为类的实例
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
    )

    data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = EventChatModel(config=config, model_args=model_args)

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=data_module['train_dataset'],
        data_collator=data_module['data_collator'],
    )
    
    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()