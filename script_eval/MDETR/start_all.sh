#!/bin/bash
# 调用 TransVG 多数据集评估脚本

IMGSIZE=224
BATCHSIZE=24
MODALITY=rgb
CUDADEVICES=3

stdbuf -oL -eL bash ./script_eval_mixup_all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg/$MODALITY/mixup_eval_all.log
