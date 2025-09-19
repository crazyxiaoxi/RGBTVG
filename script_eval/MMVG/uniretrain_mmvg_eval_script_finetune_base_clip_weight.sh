#!/bin/bash 
# flir train from clip weight
DATA_SET="rgbtvg_mixup"                                                                                                                                                                                                                                                       
IMGSIZE=224                                                                                                                                                                                                                                                                  
BATCHSIZE=32
MODALITY=rgbt 
#CUDADEVICES=0                                                                                                                                                                                                                                    
CUDADEVICES=0,1,2,3,4,5,6,7
MODEL_NAME=MMVG                                                                                                                                                                                                                                          
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')  
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
MASTER_PORT=28888
EVAL_OUT="./output_evluation/$DATA_SET"
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
CLIP_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224.pth"  
OUTPUT_DIR="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output"


# 定义统一的评估集列表
#EVAL_SETS=("val" "test" "testA" "testB" "testC")
EVAL_SETS=("val" "test" "testA" "testB" "testC" "flir_val" "flir_test" "flir_testA" "flir_testB" "flir_testC" "m3fd_val" "m3fd_test" "m3fd_testA" "m3fd_testB" "m3fd_testC" "mfad_val" "mfad_test" "mfad_testA" "mfad_testB" "mfad_testC" )


for eval_set in "${EVAL_SETS[@]}"; do
    "${DIST_CMD[@]}" --master_port $MASTER_PORT mmvg_eval.py  --model_name $MODEL_NAME  --eval_out_dir $EVAL_OUT  --modality $MODALITY  --num_workers 2  --batch_size $BATCHSIZE  --dataset $DATA_SET  --vl_hidden_dim 512  --imsize $IMGSIZE  --max_query_len 77  --normalize_before  --mixup_pretrain  --use_mask_loss  --save_hilora_clip  --hi_lora_stage 3  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --eval_model $OUTPUT_DIR/best_checkpoint.pth  --eval_set "$eval_set"  --output_dir $OUTPUT_DIR
done

