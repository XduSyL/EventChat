import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType 
import logging
import random
import re

class IeTmodel(nn.Module):
    def __init__(self, args, ViT_path) -> None:
        super().__init__()
        from .ImageEventEncoder import ImageEventEncoder
        # print("loading ImageEventEncoder")
        # self.ImageEventEncoder = ImageEventEncoder()
        # for name, param in self.ImageEventEncoder.named_parameters():
        #     param.requires_grad = False

        # self.ImageEventEncoder.eval()

        # print("loading TextEncoder")
        #替换为openclip并加入.eval
        #self.ViT_Base = CLIPModel.from_pretrained(ViT_path)
        self.ViT_Base = CLIPVisionModel.from_pretrained(ViT_path)
        self.clipprocessor = CLIPImageProcessor.from_pretrained(ViT_path)
        self.ViT_Base.requires_grad_(False)
        #self.ViT_Base.eval()
        #self.TextCLIPTokenizer = CLIPTokenizer.from_pretrained(ViT_path)
        #self.TextEncoder = self.ViT_Base.text_model        
        #self.TextEncoder.eval()

        # for name, param in self.TextEncoder.named_parameters():
        #     param.requires_grad = False

        for name, param in self.ViT_Base.named_parameters():
            param.requires_grad = False

        self.mlp_depth = args.mlp_depth
        self.visual_hidden_size = args.visual_hidden_size
        self.text_hidden_size = args.text_hidden_size
        self.hidden_size = args.LLM_hidden_size
        self.max_txt_len = args.max_txt_len
        self.end_sym= '\n'
        
        self.visual_linear = nn.Linear(self.visual_hidden_size, self.text_hidden_size)
        for name, param in self.visual_linear.named_parameters():
            param.requires_grad = False

        #self.cross_attention = CrossAttention(self.text_hidden_size)
        self.visual_projecotr = self.build_mlp_projector(self.hidden_size)
        if args.use_mm_projector:
            pretrained_weights = torch.load(args.mm_projector_path)
            pretrained_weights = {k.replace('visual_projector.', ''): v for k, v in pretrained_weights.items()}
            self.visual_projecotr.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")

        # 初始化没有从本地载入的层
        self._initialize_weights(args.use_mm_projector)

        # load LLM
        print("Start Loading LLAMA")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        #
        self.llama_model_without_lora = LlamaForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.float16,
        )

        self.llama_model = self.llama_model_without_lora

        self.llama_model.eval()

        # 冻结LLM其他参数
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        if args.useLora:
            # 配置LoRA参数
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,  # 低秩矩阵的秩
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # 列出所有你希望应用LoRA的模块
            )
            # 将LoRA应用到LLM模型上
            self.llama_model = get_peft_model(self.llama_model_without_lora, lora_config)

            # 解冻LoRA参数
            for name, param in self.llama_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

        print('Loading LLAMA Done')

    #参考LLaVA的线性层初始化方式
    def _initialize_weights(self, use_mm_projector):
        # 初始化 visual_linear 层
        print("xavier init visual_linear")
        if self.visual_linear.weight.dim() >= 2:
            init.xavier_uniform_(self.visual_linear.weight)
        if self.visual_linear.bias is not None:
            init.constant_(self.visual_linear.bias, 0)

        # 初始化 CrossAttention 和 MLP Projector 层
        #for module in [self.cross_attention, self.visual_projecotr]:
        if use_mm_projector == False:
            print("xavier init mm_projector")
            for module in [self.visual_projecotr]:
                for name, param in module.named_parameters():
                    if param.dim() >= 2:
                        init.xavier_uniform_(param)
                    elif param.dim() == 1:
                        init.constant_(param, 0)
    
    def build_mlp_projector(self, hidden_dim):
        mlp_depth = self.mlp_depth
        modules = [nn.Linear(self.text_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)

    def sample_sequence(self, logits, temperature=1.0, top_k=0, top_p=0.9):
        """
        从模型的logits中一次性采样生成一个序列。

        :param logits: 模型的输出logits, 形状为 [batch_size, sequence_length, vocab_size]
        :param temperature: 温度系数, 控制采样的随机性
        :param top_k: 如果top_k > 0, 只从top_k个最高概率的词中采样
        :param top_p: 如果top_p < 1.0, 只从累积概率小于top_p的词中采样 (nucleus sampling)
        :return: 生成的token id序列, 形状为 [batch_size, sequence_length]
        """
        logits = logits / temperature  # 应用温度系数
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 使用top_k截断logits
        if top_k > 0:
            top_k = min(top_k, sorted_logits.size(-1))
            sorted_logits = sorted_logits[..., :top_k]
            sorted_indices = sorted_indices[..., :top_k]
        
        # 使用top_p截断logits
        if top_p < 1.0:
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits[sorted_indices_to_remove] = -float('Inf')
        
        # 采样
        probs = F.softmax(sorted_logits, dim=-1)
        next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(logits.size()[:-1])
        next_tokens = sorted_indices.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        return next_tokens
        
    def forward(self, aps, dvs, video_id, Query, Answer):
        # fused_feature = self.ImageEventEncoder(aps, dvs, video_id)
        # fused_feature = self.visual_linear(fused_feature)

        #参考一下LLaVA的写法
        with torch.no_grad():
            aps = aps.tensors
            aps = aps.half()
            inputs = self.clipprocessor(images=aps, return_tensors="pt").to(aps.device)
            inputs = inputs.half()
            outputs = self.ViT_Base.vision_model(**inputs)
            fused_feature = outputs.last_hidden_state
            #fused_feature = self.visual_linear(fused_feature)           
            #fused_feature = self.ViT_Base.visual_projection(fused_feature)

        atts_img = torch.ones(fused_feature.size()[:-1], dtype=torch.long).to(fused_feature.device)

        # task_prompt = ['###Human: <Img><ImageHere></Img> '+ t for t in Query]
        task_prompt = ['###Human: <Img><ImageHere></Img> '+ t for t in Query]
        #task_prompt = '###Human: <Img><ImageHere></Img> ' + Query
        # Text_input = self.TextCLIPTokenizer(task_prompt,
        #                                      return_tensors="pt", padding=True)
        # Text_input = {key: value.to('cuda') for key, value in Text_input.items()}
        # text_features = self.TextEncoder(**Text_input).last_hidden_state       
        #cross_attn_output = self.cross_attention(fused_feature, text_features)
        
        visual_to_text = self.visual_projecotr(fused_feature)

        #Image concat Text
        img_embeds, atts_img = self.prompt_wrap(visual_to_text, atts_img, task_prompt)

        description = [""]*len(Query)
        text_input = [d + a for (d,a) in zip(description, Answer)]

        text = [t + self.end_sym for t in text_input]

        #llm_input = Query + Answer
        self.llama_tokenizer.padding_side = "right"

        #text是labels，也就是需要通过LLM生成的目标文本
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(fused_feature.device)
        
        #target将padding出来的变成-100
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        #为了和输入序列的长度对齐
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(fused_feature.device).fill_(-100)
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with torch.cuda.amp.autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        logits = outputs.logits  # 获取logits

        # 使用argmax得到每个样本最有可能的词ID
        predicted_ids = torch.argmax(logits, dim=-1)

        tokenizer = self.llama_tokenizer

        # 将每个样本的词ID转换为实际的文本
        predicted_texts = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        predicted_texts = [text.strip().replace('\n', ' ').replace('  ', ' ') for text in predicted_texts]

        predicted_texts = [re.sub(r'[^\w\s.,]', '', text) for text in predicted_texts]

        # 创建或打开一个txt文件以写入预测文本
        with open('predicted_texts.txt', 'a', encoding='utf-8') as f:
            for i, text in enumerate(predicted_texts):
                # 10%的概率写入文本
                if random.random() < 0.1:
                    f.write(f"Sample {i + 1}: {text}\n")

        return loss
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            if type(prompt) is str:
                batch_size = img_embeds.shape[0]
                p_before, p_after = prompt.split('<ImageHere>')
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embeds = self.llama_model_without_lora.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model_without_lora.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
                wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
                return wrapped_img_embeds, wrapped_atts_img
            else:
                batch_size = img_embeds.shape[0]
                prompt_splitted = [p.split('<ImageHere>') for p in prompt]
                p_before, p_after = [x[0] for x in prompt_splitted], [x[1] for x in prompt_splitted]
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False, padding="longest", truncation=True,).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False,  padding="longest", truncation=True,).to(img_embeds.device)
                #通过LLM的Embedding层对文本进行嵌入
                p_before_embeds = self.llama_model_without_lora.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model_without_lora.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
                wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
                return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, img_features, txt_features):
        attn_output, _ = self.multihead_attn(query=img_features, key=txt_features, value=txt_features)
        attn_output = self.layer_norm(attn_output + img_features)
        return attn_output