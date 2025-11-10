#!/bin/bash

IMGSIZE=224

CUDADEVICES=0,1
EPOCHS=120
echo -e "\n\n===================== 启动全部训练与测试 ====================="
BATCHSIZE=1
MODALITY='rgbt'
bash ./script_train/OneRef_large_rec/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS


echo -e "\n\n===================== 所有任务已执行完毕 =====================
