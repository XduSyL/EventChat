from lightning_trainer import get_args_parser
import argparse
import torch
from SODFormer.models import build_spatial, build_temporal, build_fusion, build_cri_pro
import SODFormer.util.misc as utils
import SODFormer.datasets.samplers as samplers
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ImageEventEncoder():
    parser = argparse.ArgumentParser('SODFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    image_event_encoder = build_fusion(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    missing_keys, unexpected_keys = image_event_encoder.load_state_dict(checkpoint['model'], strict=False)
    
    return image_event_encoder