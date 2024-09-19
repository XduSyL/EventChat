import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset.IeTdataset import DSECDet, collate_fn, make_transforms
from model.IeTGPT import IeTmodel
import logging
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.common.optim import LinearWarmupCosineLRScheduler
import warnings
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

# 忽略所有警告
warnings.filterwarnings("ignore")
logger = TensorBoardLogger("tb_logs", name="IeTGPT")

class SODFormerModule(pl.LightningModule):
    def __init__(self, args):
        super(SODFormerModule, self).__init__()
        self.args = args
        self.model = IeTmodel(args, ViT_path=args.ViT_path)
        self.lr_scheduler = None  # Placeholder for lr_scheduler
        
    def forward(self, aps, dvs, video_id, Query, Answer):
        return self.model(aps, dvs, video_id, Query, Answer)

    def training_step(self, batch, batch_idx):
        img = batch['img'].to(self.args.device)
        events = batch['events'].to(self.args.device)
        query = batch['Query']
        answer = batch['Answer']
        
        # 获取当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        loss = self.model(img, events, batch['dataset_idx'][0], query, answer)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if isinstance(generated_text, list):
        #     generated_text = "\n".join(generated_text)  # Join list into a single string

        # if batch_idx % 4 == 0:  # Modify the condition as needed
        #     self.logger.experiment.add_text(f"generated_text/step_{batch_idx}", generated_text, global_step=batch_idx)
        return loss

    def configure_optimizers(self):
        # 如果使用DeepSpeed，DeepSpeed会自动处理优化器，无需手动返回
        if self.trainer.strategy_name == "deepspeed":
            return None
        
        optimizer = self.build_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epoch)
        return [optimizer], [scheduler]

    def build_optimizer(self):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        logging.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.optimizer_weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.optimizer_init_lr),
            weight_decay=float(self.args.optimizer_weight_decay)
        )
        return optimizer

    def build_lr_scheduler(self, optimizer):
        lr_sched = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=self.args.max_epoch,
            iters_per_epoch=self.args.iters_per_epoch,
            min_lr=self.args.min_lr,
            init_lr=self.args.init_lr,
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )
        return lr_sched


def get_args_parser():
    parser = argparse.ArgumentParser('SODFormer Detector', add_help=False)

    # Data Loading
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--cache_mode', default=True, action='store_true', help='whether to cache images on memory')

    # Setup
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # Variants of SODFormer
    parser.add_argument('--no_temporal', default=False, action='store_true',
                        help='If true, single frame detection will be implemented')
    parser.add_argument('--no_frame', default=False, action='store_true',
                        help='If true, only event stream will be used')
    parser.add_argument('--no_event', default=False, action='store_true',
                        help='If true, only frame stream will be used')
    parser.add_argument('--event_repre', default='image', type=str, choices=['image', 'voxel', 'gray'],
                        help='Event representation of event stream, disabled when no_event is set True')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the Spatial Transformer Encoder")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the Spatial Transformer Decoder")
    parser.add_argument('--tem_enc_layers', default=6, type=int,
                        help="Number of encoding layers in the Temporal Transformer Encoder")
    parser.add_argument('--tem_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the Temporal Transformer Decoder")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int, help='Number of sampling points in decoders')
    parser.add_argument('--enc_n_points', default=4, type=int, help='Number of sampling points in encoders')
    parser.add_argument('--n_frames', default=8, type=int, help='Temporal aggregation size')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of categories + 1')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # Dataset paths
    parser.add_argument('--frame_path', default='./data/aps_frames', type=str)
    parser.add_argument('--anno_path', default='./data/annotations', type=str)
    parser.add_argument('--event_path', default='./data/events_npys', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--categories_path', default='./datasets/neuromorphic_object_classes.txt')
    parser.add_argument('--spatial_aps_model', default='')
    parser.add_argument('--spatial_dvs_model', default='')
    parser.add_argument('--temporal_aps_model', default='')
    parser.add_argument('--temporal_dvs_model', default='')

    # Output paths
    parser.add_argument('--output_dir', default='./results/models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_method', default='SODFormer')
    parser.add_argument('--eval_file', default='SODFormer_eval.pth')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')

    # Dataset
    parser.add_argument('--dsec_path', type=Path, default='/data/SyL/Event_RGB/dataset/dsec-dataset')

    # Pre-train path
    parser.add_argument('--ViT_path', default='/data/SyL/model/clip-vit-large-patch14-336')
    parser.add_argument('--llama_model', default='/data/SyL/model/vicuna-7b-v1.5/')

    # model
    parser.add_argument('--mlp_depth', default=2, type=int)
    parser.add_argument('--visual_hidden_size', default=768, type=int)
    parser.add_argument('--text_hidden_size', default=1024, type=int)
    parser.add_argument('--LLM_hidden_size', default=4096, type=int)

    #
    parser.add_argument('--max_txt_len', default=110, type=int)
    parser.add_argument('--useLora', default=False, type=bool)

    # optimizer
    parser.add_argument('--optimizer_weight_decay', default=0.05, type=float)
    parser.add_argument('--optimizer_init_lr', default=1e-4, type=float)

    # lr
    parser.add_argument('--init_lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup_lr', default=1e-4, type=float)
    parser.add_argument('--iters_per_epoch', default=200, type=int)
    
    # epoch
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--warmup_steps', default=200, type=int)

    # gpus
    parser.add_argument('--gpus', type=int, required=True)

    # mm_projector
    parser.add_argument('--use_mm_projector', default=True, type=int)
    parser.add_argument('--mm_projector_path', default='/data/SyL/model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mlp_weights.pth')

    

    return parser

def build_dataloader(args, batch_size):
    dataset = DSECDet(args.dsec_path, batch_size=batch_size, split="val", sync="back", debug=False, transform=make_transforms())
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
    return dataloader_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SODFormer Detector', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args.batch_size)

    # Initialize model
    model = SODFormerModule(args)

    # Build dataloader
    dataloader_train = build_dataloader(args, args.batch_size)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,  # 保存所有检查点，如果只保存最佳，可以设置为1
        every_n_epochs=50,  # 每隔5个epoch保存一次
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        gpus=args.gpus if torch.cuda.is_available() else 0,  # 根据需要调整GPU数量
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.clip_max_norm,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=4,
        strategy=DeepSpeedStrategy(config="/data/SyL/Event_RGB/deepspeed_config.json"),
        precision=16
    )

    # Training    
    trainer.fit(model, train_dataloaders=dataloader_train)