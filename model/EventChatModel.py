import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
from transformers import LlamaTokenizer, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType 
from dataset.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from typing import List, Optional, Tuple, Union, Dict, Callable
from transformers.generation.utils import GenerateOutput
import logging
import random
import re
import os
import json
import pdb

class EventChatConfig(LlamaConfig):
    model_type = "EventChat_llama" 


class VisualTower(nn.Module):
    def __init__(self, visual_tower):
        super().__init__()

        self.visual_tower_name = visual_tower
        self.image_processor = CLIPImageProcessor.from_pretrained(self.visual_tower_name)
        self.visual_tower = CLIPVisionModel.from_pretrained(self.visual_tower_name)
        self.visual_tower.requires_grad_(False)
    
    def forward(self, image_tensor):
        outputs = self.visual_tower.vision_model(image_tensor)
        images_feature = outputs.last_hidden_state
        images_feature = self.visual_projecotor(images_feature)

        return images_feature



class EventChatLlamaModel(LlamaModel):
    config_class = EventChatConfig

    def __init__(self, config: LlamaConfig):
        super(EventChatLlamaModel, self).__init__(config)

        self.mlp_depth = 2
        self.text_hidden_size = 1024
        self.hidden_size = 4096 
        if hasattr(config, "mm_visual_tower"):
            self.visual_tower = self.build_visual_tower(config.mm_visual_tower)
            self.visual_projector = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.float16)

    def build_visual_tower(self, visual_tower):
        return VisualTower(visual_tower)
        

    def build_mlp_projector(self, text_hidden_size, hidden_dim):
        mlp_depth = self.mlp_depth
        modules = [nn.Linear(text_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        visual_tower = model_args.vision_tower
        self.config.mm_visual_tower = visual_tower

        visual_tower = self.build_visual_tower(model_args.vision_tower) 

        self.visual_tower = visual_tower

        self.visual_projector = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.float16)

        if model_args.pretrain_mm_mlp_adapter is not None:
            print("loading mm_projector pretrain")
            pretrained_weights = torch.load(model_args.pretrain_mm_mlp_adapter)
            pretrained_weights = {k.replace("model.visual_projector.", ""): v for k, v in pretrained_weights.items()}
            self.visual_projector.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")


class EventChatModel(LlamaForCausalLM):

    config_class = EventChatConfig

    def __init__(self, config) -> None:
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = EventChatLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_visual_tower(self):
        return self.get_model().visual_tower
    
    def visval_encode(self, image_tensor):
        outputs = self.get_model().visual_tower.visual_tower(image_tensor)
        images_feature = outputs.last_hidden_state
        images_feature = self.get_model().visual_projector(images_feature)

        return images_feature


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    
    def forward(self, 
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None, 
            labels: Optional[torch.LongTensor] = None,
            events: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            image_sizes: Optional[List[List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
    
        if inputs_embeds is None:
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
                image_sizes           
            )
            
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
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_feature = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            raise NotImplementedError("please input Image")
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    #     print("loading CLIP Tower")
    #     self.CLIPVisionModel = CLIPVisionModel.from_pretrained(model_args.vision_tower, torch_dtype=torch.float16)
    #     self.CLIPProcessor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)
    #     self.CLIPVisionModel.requires_grad_(False)
        
    #     self.mlp_depth = 2
    #     self.text_hidden_size = 1024
    #     self.hidden_size = 4096       
    #     if model_args.pretrain_mm_mlp_adapter:
    #         print("loading mm_projector pretrain")
    #         self.visual_projecotor = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.float16)
    #         pretrained_weights = torch.load(model_args.pretrain_mm_mlp_adapter)
    #         pretrained_weights = {k.replace('visual_projecotor.', ''): v for k, v in pretrained_weights.items()}
    #         self.visual_projecotor.load_state_dict(pretrained_weights, strict=True)
    #         print("Pretrained weights loaded successfully into visual_projector.")
    #     else:
    #         self.visual_projecotor = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.float16) 

    #     #self.cross_attention = CrossAttention(self.text_hidden_size)

    #     # 初始化没有从本地载入的层
    #     self._initialize_weights(getattr(model_args, 'use_mm_projector', False))

    #     # load LLM
    #     print("Start Loading LLAMA")
    #     self.llama_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=2048,
    #                                                          padding_side="right")
    #     self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
    #     #self.llama_tokenizer.eos_token = '<eos>'

        
    #     self.llama_model = LlamaForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         torch_dtype=torch.float16,
    #     )
    #     self.model = self.llama_model.model
    #     self.lm_head = self.llama_model.lm_head
        
    #     # 冻结LLM其他参数
    #     for name, param in self.llama_model.named_parameters():
    #         param.requires_grad = False

    #     if model_args.useLorafinetune:
    #         # 配置LoRA参数
    #         lora_config = LoraConfig(
    #             task_type=TaskType.CAUSAL_LM,
    #             inference_mode=False,
    #             r=16,  # 低秩矩阵的秩
    #             lora_alpha=32,
    #             lora_dropout=0.1,
    #             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # 列出所有你希望应用LoRA的模块
    #         )
    #         # 将LoRA应用到LLM模型上
    #         self.llama_model = get_peft_model(self.llama_model, lora_config)

    #         # 解冻LoRA参数
    #         for name, param in self.llama_model.named_parameters():
    #             if 'lora' in name:
    #                 param.requires_grad = True

    #     #print('Loading LLAMA Done')

    #     # 打印可训练参数的详细信息
    #     total_trainable_params = 0
    #     print("可训练的参数详情：")
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             param_count = param.numel()
    #             total_trainable_params += param_count
    #             print(f"{name}: shape={param.shape}, 参数数量={param_count}")
    #     print(f"总可训练参数数量：{total_trainable_params}")

    # #参考LLaVA的线性层初始化方式
    # def _initialize_weights(self, use_mm_projector):
    #     # 初始化 visual_linear 层
    #     # print("xavier init visual_linear")
    #     # if self.visual_linear.weight.dim() >= 2:
    #     #     init.xavier_uniform_(self.visual_linear.weight)
    #     # if self.visual_linear.bias is not None:
    #     #     init.constant_(self.visual_linear.bias, 0)

    #     # 初始化 CrossAttention 和 MLP Projector 层
    #     #for module in [self.cross_attention, self.visual_projecotor]:
    #     if use_mm_projector == True:
    #         print("xavier init mm_projector")
    #         for module in [self.visual_projecotor]:
    #             for name, param in module.named_parameters():
    #                 if param.dim() >= 2:
    #                     init.xavier_uniform_(param)
    #                 elif param.dim() == 1:
    #                     init.constant_(param, 0)
   
    # def build_mlp_projector(self, text_hidden_size, hidden_dim):
    #     mlp_depth = self.mlp_depth
    #     modules = [nn.Linear(text_hidden_size, hidden_dim)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(hidden_dim, hidden_dim))
    #     return nn.Sequential(*modules)
    
    # def save_model(self, output_dir=None):
    #     if output_dir is None:
    #         output_dir = self.model_args.output_mm_mlp_adapter

    #     # 分别收集 LoRA 参数和视觉投影器参数
    #     lora_state_dict = {}
    #     visual_projecotor_state_dict = {}

    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             if 'lora_' in name:
    #                 lora_state_dict[name] = param.cpu()
    #             elif 'visual_projecotor' in name:
    #                 visual_projecotor_state_dict[name] = param.cpu()

    #     # 创建输出目录（如果不存在）
    #     os.makedirs(output_dir, exist_ok=True)

    #     # 如果使用 LoRA 微调，保存 LoRA 参数和配置
    #     if self.model_args.useLorafinetune:
    #         # 保存 LoRA 参数
    #         lora_save_path = os.path.join(output_dir, 'lora_parameters.pth')
    #         torch.save(lora_state_dict, lora_save_path)
    #         print(f"LoRA 参数已保存到 {lora_save_path}")

    #         # 保存 LoRA 配置文件
    #         lora_config = LoraConfig(
    #             task_type=TaskType.CAUSAL_LM,
    #             r=16,
    #             lora_alpha=32,
    #             lora_dropout=0.1,
    #             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    #         )
    #         lora_config.save_pretrained(output_dir)
    #         print(f"LoRA 配置已保存到 {output_dir}")
        
    #     # 保存视觉投影器参数
    #     visual_save_path = os.path.join(output_dir, 'visual_projecotor_parameters.pth')
    #     torch.save(visual_projecotor_state_dict, visual_save_path)
    #     print(f"视觉投影器参数已保存到 {visual_save_path}")

    # def visval_encode(self, image_tensor):
    #     outputs = self.CLIPVisionModel.vision_model(image_tensor)
    #     images_feature = outputs.last_hidden_state
    #     images_feature = self.visual_projecotor(images_feature)

    #     return images_feature

    # def save_pretrained(
    #     self,
    #     save_directory: Union[str, os.PathLike],
    #     is_main_process: bool = True,
    #     state_dict: Optional[Dict[str, torch.Tensor]] = None,
    #     save_function: Callable = torch.save,
    #     push_to_hub: bool = False,
    #     max_shard_size: Union[int, str] = None,
    #     **kwargs,
    # ):
    #     # 调用自定义的保存方法，传递必要的参数
    #     self.save_model(save_directory)
      
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):  
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        #对图像通过visual encoder进行编码并且对齐到LLM的隐藏层维度
        image_features = self.visval_encode(images)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        #将attention_mask转换成bool类型
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        #去掉padding出来的所有元素，只保留有用的部分
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            #检查输入中图像标志的数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            #若输入中不包含图像，则为纯文本问题
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            #若输入中包含有图像
            #找出包含的图像标记的索引位置，cur_input_ids.shape[0]表示序列长度
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            #从输入中分离出不包含图像的文本的部分，及其标签
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #通过LLM对input进行embed
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            #本段用于处理将文本token和图像token拼接起来,并把图像token的部分在label中都是IGNORE_INDEX进行padding
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
        #此处是通过最大长度做截断处理
        tokenizer_model_max_length = 2048
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        #max_len计算一个批次中的最大长度，为做批处理作出准备
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        #此处会将所有标签都padding成为该batch中样本最长的那个长度
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        #首先将attention_mask和position_ids全部初始化为全零向量
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            #遍历每个样本的嵌入
            cur_len = cur_new_embed.shape[0]
            # if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            #默认情况下项右进行padding
            #将输入嵌入都padding为当前批次中最大的长度
            #并设置注意力掩码，将padding出来的都设置为false
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

        #将输入嵌入打包成tensor
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
      
AutoConfig.register("EventChat_llama", EventChatConfig)
AutoModelForCausalLM.register(EventChatConfig, EventChatModel)


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, img_features, txt_features):
        attn_output, _ = self.multihead_attn(query=img_features, key=txt_features, value=txt_features)
        attn_output = self.layer_norm(attn_output + img_features)
        return attn_output