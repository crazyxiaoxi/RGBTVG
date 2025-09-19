#!/bin/bash 
# flir train from clip weight
DATA_SET="rgbtvg_mixup"                                                                                                                                                                                                                                                       
IMGSIZE=224                                                                                                                                                                                                                                                                  
BATCHSIZE=32
MODALITY=rgbt 
# CUDADEVICES=0                                                                                                                                                                                                                                    
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
OUTPUT_DIR_WARMUP="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v100"
OUTPUT_DIR_STAGE1="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v101"
OUTPUT_DIR_STAGE2="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v102"
OUTPUT_DIR_STAGE3="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v103"


# 定义统一的评估集列表
#EVAL_SETS=("val" "test" "testA" "testB" "testC")
EVAL_SETS=("val" "test" "testA" "testB" "testC" "flir_val" "flir_test" "flir_testA" "flir_testB" "flir_testC" "m3fd_val" "m3fd_test" "m3fd_testA" "m3fd_testB" "m3fd_testC" "mfad_val" "mfad_test" "mfad_testA" "mfad_testB" "mfad_testC" )

# Warmup 阶段评估
#for eval_set in "${EVAL_SETS[@]}"; do
#    "${DIST_CMD[@]}" --master_port $MASTER_PORT mmvg_eval.py --model_name $MODEL_NAME --eval_out_dir $EVAL_OUT  --modality $MODALITY  --num_workers 2  --batch_size $BATCHSIZE  --dataset $DATA_SET  --vl_hidden_dim 512  --imsize $IMGSIZE  --max_query_len 77  --normalize_before   --use_mask_loss  --save_hilora_clip --mixup_pretrain  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth  --eval_set "$eval_set"  --output_dir $OUTPUT_DIR_WARMUP
#done

# Stage 1 评估
#for eval_set in "${EVAL_SETS[@]}"; do
#    "${DIST_CMD[@]}" --master_port $MASTER_PORT mmvg_eval.py  --model_name $MODEL_NAME  --eval_out_dir $EVAL_OUT  --modality $MODALITY  --num_workers 2  --batch_size $BATCHSIZE  --dataset $DATA_SET  --vl_hidden_dim 512  --imsize $IMGSIZE  --max_query_len 77  --normalize_before --mixup_pretrain  --use_mask_loss  --save_hilora_clip  --hi_lora_stage 1  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth  --eval_set "$eval_set"  --output_dir $OUTPUT_DIR_STAGE1
#done

#Stage 2 评估
#for eval_set in "${EVAL_SETS[@]}"; do
#   "${DIST_CMD[@]}" --master_port $MASTER_PORT mmvg_eval.py  --model_name $MODEL_NAME  --eval_out_dir $EVAL_OUT  --modality $MODALITY  --num_workers 2  --batch_size $BATCHSIZE  --dataset $DATA_SET  --vl_hidden_dim 512  --imsize $IMGSIZE  --max_query_len 77  --normalize_before  --mixup_pretrain  --use_mask_loss  --save_hilora_clip  --hi_lora_stage 2  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth  --eval_set "$eval_set"  --output_dir $OUTPUT_DIR_STAGE2
#done

# Stage 3 评估
for eval_set in "${EVAL_SETS[@]}"; do
    "${DIST_CMD[@]}" --master_port $MASTER_PORT mmvg_eval.py  --model_name $MODEL_NAME  --eval_out_dir $EVAL_OUT  --modality $MODALITY  --num_workers 2  --batch_size $BATCHSIZE  --dataset $DATA_SET  --vl_hidden_dim 512  --imsize $IMGSIZE  --max_query_len 77  --normalize_before  --mixup_pretrain  --use_mask_loss  --save_hilora_clip  --hi_lora_stage 3  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth  --eval_set "$eval_set"  --output_dir $OUTPUT_DIR_STAGE3
done


# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set val      --output_dir $OUTPUT_DIR_WARMUP;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set test     --output_dir $OUTPUT_DIR_WARMUP;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set testA    --output_dir $OUTPUT_DIR_WARMUP;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set testB    --output_dir $OUTPUT_DIR_WARMUP;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set testC    --output_dir $OUTPUT_DIR_WARMUP;


# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE1;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE1;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set testA  --output_dir $OUTPUT_DIR_STAGE1;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set testB  --output_dir $OUTPUT_DIR_STAGE1;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set testC  --output_dir $OUTPUT_DIR_STAGE1;

# #
# # stage 2
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE2;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE2;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set testA  --output_dir $OUTPUT_DIR_STAGE2;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set testB  --output_dir $OUTPUT_DIR_STAGE2;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set testC  --output_dir $OUTPUT_DIR_STAGE2;

# #
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE3;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE3;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set testA  --output_dir $OUTPUT_DIR_STAGE3;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set testB  --output_dir $OUTPUT_DIR_STAGE3;
# "${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --eval_out_dir $EVAL_OUT --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set testC  --output_dir $OUTPUT_DIR_STAGE3;
