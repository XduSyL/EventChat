import logging
import os
from common.logger import MetricLogger, SmoothedValue
import torch
import cv2
import numpy as np

def train_one_epoch(args,
                    epoch,
                    iters_per_epoch,
                    model,
                    data_loader,
                    optimizer,
                    lr_scheduler,
                    scaler=None,
                    log_freq=50,
                    accum_grad_iters=1):

    # Log config
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

    use_amp = scaler is not None
    use_amp = False

    logging.info(
    "Start training epoch {}, {} iters per inner epoch.".format(
        epoch, iters_per_epoch
        )
    )
    header = "Train: data epoch: [{}]".format(epoch)

    inner_epoch = epoch

    if epoch == 3:
        print(1)

    for i, samples in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        samples.update(
            {
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            }
        )

        lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

        with torch.cuda.amp.autocast(enabled=use_amp):
            item_i = 0
            query = samples['Query']
            answer = samples['Answer']
            img = samples['img']

            # #调试
            # imgage_tensors = img.tensors
            # # 提取第一张图像
            # first_img = imgage_tensors[0].cpu().detach().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组，并调整维度顺序
            # first_img = (first_img * 255).astype(np.uint8)  # 缩放到 [0, 255] 并转换为 uint8 类型

            # # 使用 OpenCV 保存第一张图像
            # cv2.imwrite("first_img.png", first_img)

            events = samples['events']
            targets = samples['target']
            img = img.to(args.device)
            events = events.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            loss, generate_text = model(img, events, samples['dataset_idx'][0], query, answer)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (item_i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

                item_i += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        logging.info(f"Generated Text\n: {generate_text}")

    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }