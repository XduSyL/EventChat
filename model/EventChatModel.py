import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType 
from dataset.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from typing import List, Optional, Tuple, Union, Dict, Callable
import logging
import random
import re
import os
import json

class EventChatModel(LlamaForCausalLM):
    def __init__(self, config, model_args) -> None:
        super().__init__(config)
        #super().__init__(config)
        #from .ImageEventEncoder import ImageEventEncoder
        # print("loading ImageEventEncoder")    
        # self.ImageEventEncoder = ImageEventEncoder()
        # for name, param in self.ImageEventEncoder.named_parameters():
        #     param.requires_grad = False

        # self.ImageEventEncoder.eval()
        
        #self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.model_args = model_args
        
        print("loading CLIP Tower")
        self.CLIPVisionModel = CLIPVisionModel.from_pretrained(model_args.vision_tower)
        self.CLIPProcessor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)
        self.CLIPVisionModel.requires_grad_(False)
        
        self.mlp_depth = 2
        self.text_hidden_size = 1024
        self.hidden_size = 4096       
        if model_args.pretrain_mm_mlp_adapter:
            print("loading mm_projector pretrain")
            self.visual_projecotor = self.build_mlp_projector(self.text_hidden_size, self.hidden_size)  
            pretrained_weights = torch.load(model_args.pretrain_mm_mlp_adapter)
            pretrained_weights = {k.replace('visual_projecotor.', ''): v for k, v in pretrained_weights.items()}
            self.visual_projecotor.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")
        else:
            self.visual_projecotor = self.build_mlp_projector(self.text_hidden_size, self.hidden_size)  

        #self.cross_attention = CrossAttention(self.text_hidden_size)

        # 初始化没有从本地载入的层
        self._initialize_weights(getattr(model_args, 'use_mm_projector', False))

        # load LLM
        print("Start Loading LLAMA")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=2048,
                                                             padding_side="right")
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        
        self.llama_model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.model = self.llama_model.model
        self.lm_head = self.llama_model.lm_head
        
        # 冻结LLM其他参数
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        if model_args.useLorafinetune:
            # 配置LoRA参数
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # 低秩矩阵的秩
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # 列出所有你希望应用LoRA的模块
            )
            # 将LoRA应用到LLM模型上
            self.llama_model = get_peft_model(self.llama_model, lora_config)

            # 解冻LoRA参数
            for name, param in self.llama_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

        #print('Loading LLAMA Done')

        # 打印可训练参数的详细信息
        total_trainable_params = 0
        print("可训练的参数详情：")
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_trainable_params += param_count
                print(f"{name}: shape={param.shape}, 参数数量={param_count}")
        print(f"总可训练参数数量：{total_trainable_params}")

    #参考LLaVA的线性层初始化方式
    def _initialize_weights(self, use_mm_projector):
        # 初始化 visual_linear 层
        # print("xavier init visual_linear")
        # if self.visual_linear.weight.dim() >= 2:
        #     init.xavier_uniform_(self.visual_linear.weight)
        # if self.visual_linear.bias is not None:
        #     init.constant_(self.visual_linear.bias, 0)

        # 初始化 CrossAttention 和 MLP Projector 层
        #for module in [self.cross_attention, self.visual_projecotor]:
        if use_mm_projector == True:
            print("xavier init mm_projector")
            for module in [self.visual_projecotor]:
                for name, param in module.named_parameters():
                    if param.dim() >= 2:
                        init.xavier_uniform_(param)
                    elif param.dim() == 1:
                        init.constant_(param, 0)
   
    def build_mlp_projector(self, text_hidden_size, hidden_dim):
        mlp_depth = self.mlp_depth
        modules = [nn.Linear(text_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    
    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.model_args.output_mm_mlp_adapter

        # 分别收集 LoRA 参数和视觉投影器参数
        lora_state_dict = {}
        visual_projecotor_state_dict = {}

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_state_dict[name] = param.cpu()
                elif 'visual_projecotor' in name:
                    visual_projecotor_state_dict[name] = param.cpu()

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 保存 LoRA 参数
        lora_save_path = os.path.join(output_dir, 'lora_parameters.pth')
        torch.save(lora_state_dict, lora_save_path)
        print(f"LoRA 参数已保存到 {lora_save_path}")

        # 保存视觉投影器参数
        visual_save_path = os.path.join(output_dir, 'visual_projecotor_parameters.pth')
        torch.save(visual_projecotor_state_dict, visual_save_path)
        print(f"视觉投影器参数已保存到 {visual_save_path}")

        # 保存 LoRA 配置文件
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        )
        lora_config.save_pretrained(output_dir)
        print(f"LoRA 配置已保存到 {output_dir}")


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = None,
        **kwargs,
    ):
        # 调用自定义的保存方法，传递必要的参数
        self.save_model(save_directory)
      
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_features, image_sizes=None
    ): 
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.llama_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #通过LLM对input进行embed
            cur_input_embeds = self.llama_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        #tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        tokenizer_model_max_length = 2048
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            tokenizer_padding_side = 'right'
            if  tokenizer_padding_side == 'left':
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
     
    def forward(self, images, events, input_ids, labels, 
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                image_sizes: Optional[List[List[int]]] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        # fused_feature = self.ImageEventEncoder(aps, dvs, video_id)
        # fused_feature = self.visual_linear(fused_feature)

        #inputs_images = self.CLIPProcessor(images=images, return_tensors="pt").to(images.device)

        outputs = self.CLIPVisionModel.vision_model(images)
        images_feature = outputs.last_hidden_state
        images_feature = self.visual_projecotor(images_feature)
        
        # self.device = images.device

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            images_feature,
            image_sizes           
        )
        # with torch.cuda.amp.autocast(): 
        #     outputs = self.llama_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         labels=labels,
        #         use_cache=use_cache,
        #         past_key_values=past_key_values,
        #         position_ids=position_ids,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #         return outputs
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
     
        return outputs
    
  
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, img_features, txt_features):
        attn_output, _ = self.multihead_attn(query=img_features, key=txt_features, value=txt_features)
        attn_output = self.layer_norm(attn_output + img_features)
        return attn_output