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
from dataset.IeTdataset import DSECDet
from dataset.IeTdataset import collate_fn, make_transforms
from model.IeTGPT import IeTmodel
import logging
from model.common.optim import LinearWarmupCosineLRScheduler
from runner.engine import train_one_epoch
import warnings
from lightning_trainer import get_args_parser

logging.basicConfig(
    filename='/data/SyL/Event_RGB/log/training_log_0831.log',  # 指定日志文件的名称
    level=logging.INFO,           # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)
# 忽略所有警告
warnings.filterwarnings("ignore")

torch.manual_seed(42)

def build_optimizer(args, model):
# TODO make optimizer class and configurations
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        print(n) # print trainable parameter
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()
    logging.info("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(args.optimizer_weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    #beta2 = self.config.run_cfg.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(args.optimizer_init_lr),
        weight_decay=float(args.optimizer_weight_decay)
    )

    return optimizer

def build_lr_scheduler(args, Optimizer):
    #lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
    max_epoch = args.max_epoch
    min_lr = args.min_lr
    init_lr = args.init_lr

    # optional parameters
    decay_rate = None 
    warmup_start_lr = args.warmup_lr
    warmup_steps = args.warmup_steps
    iters_per_epoch = args.iters_per_epoch

    lr_sched = LinearWarmupCosineLRScheduler(
        optimizer=Optimizer,
        max_epoch=max_epoch,
        iters_per_epoch=iters_per_epoch,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_rate=decay_rate,
        warmup_start_lr=warmup_start_lr,
        warmup_steps=warmup_steps,
    )

    return lr_sched


if __name__ == '__main__':
    parser = argparse.ArgumentParser('IeTGPT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    model = IeTmodel(args, ViT_path=args.ViT_path)
    model.cuda()

        # 使用 DataParallel 进行多GPU并行训练
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    #载入数据
    batch_size = 2
    dataset = DSECDet(args.dsec_path, batch_size=batch_size, split="val", sync="back", debug=False, transform=make_transforms())
   
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
    optimizer = build_optimizer(args, model)
    lr_sched = build_lr_scheduler(args, optimizer)

    scaler = torch.cuda.amp.GradScaler()   
    for epoch in range(1, args.max_epoch):
        metrics = train_one_epoch(args,
                                epoch=int(epoch),
                                iters_per_epoch=args.iters_per_epoch,
                                model=model,
                                data_loader=dataloader_train,
                                optimizer=optimizer,
                                lr_scheduler=lr_sched,
                                scaler=scaler)
    
    print("train complete")