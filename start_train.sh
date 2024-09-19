#!/bin/bash

# 设置可见的GPU设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 运行 Python 脚本并传递参数
 python lightning_trainer.py  --gpus 4 --resume /data/SyL/Event_RGB/SODFormer/data/sodformer.pth --eval