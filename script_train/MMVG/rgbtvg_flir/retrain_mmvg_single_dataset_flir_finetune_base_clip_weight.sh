# flir train from clip weight
DATA_SET="rgbtvg_flir"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
MODEL_NAME=${MODEL_NAME:-MMVG}
LAVS_MODE=${LAVS_MODE:-lavs}
LORA_R_RGB=${LORA_R_RGB:-16}
LORA_R_IR=${LORA_R_IR:-48}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-60}
STAGE_EPOCHS=${STAGE_EPOCHS:-20}
RETRAIN=${RETRAIN:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth"}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
CLIP_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224.pth"  
RUN_TAG="${LAVS_MODE}/lora_r${LORA_R_RGB}_${LORA_R_IR}"
OUTPUT_DIR_WARMUP="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/${RUN_TAG}/output_v100"
OUTPUT_DIR_STAGE1="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/${RUN_TAG}/output_v101"
OUTPUT_DIR_STAGE2="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/${RUN_TAG}/output_v102"
OUTPUT_DIR_STAGE3="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/${RUN_TAG}/output_v103"

echo -e "\n\n\n\n\n\n\n==================== flir warmup ==========================="
"${DIST_CMD[@]}" --master_port 28887 train_val/mmvg_train.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 4 --epochs ${WARMUP_EPOCHS}  --batch_size $BATCHSIZE --lr 0.0005   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512  --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --dataset $DATA_SET      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss   --data_root $DATA_ROOT  --split_root $SPLIT_ROOT   --output_dir $OUTPUT_DIR_WARMUP  --retrain $RETRAIN;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set val      --output_dir $OUTPUT_DIR_WARMUP;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_WARMUP/best_checkpoint.pth --eval_set test     --output_dir $OUTPUT_DIR_WARMUP;
##
# stage 1
echo -e "\n\n\n\n\n\n\n==================== flir stage 1 ==========================="
"${DIST_CMD[@]}" --master_port 28887 train_val/mmvg_train.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 4 --epochs ${STAGE_EPOCHS}  --batch_size $BATCHSIZE --lr 0.00010   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --dataset $DATA_SET      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --mixup_pretrain --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --hi_lora_retrain $OUTPUT_DIR_WARMUP/best_checkpoint.pth      --hi_lora_clip $OUTPUT_DIR_WARMUP/clip_lora_stage_with_bridge.pth       --output_dir $OUTPUT_DIR_STAGE1/     --sup_type full;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE1;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE1;

#
# stage 2

#echo -e "\n\n\n\n\n\n\n==================== flir stage 2 ==========================="
"${DIST_CMD[@]}" --master_port 28887 train_val/mmvg_train.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 4 --epochs ${STAGE_EPOCHS}  --batch_size $BATCHSIZE --lr 0.00002   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --dataset $DATA_SET      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --mixup_pretrain --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --hi_lora_retrain $OUTPUT_DIR_STAGE1/best_checkpoint.pth      --hi_lora_clip $OUTPUT_DIR_STAGE1/clip_lora_stage_with_bridge.pth       --output_dir $OUTPUT_DIR_STAGE2/     --sup_type full;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE2;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE2;

##
## stage 3
#echo -e "\n\n\n\n\n\n\n==================== flir stage 3 ==========================="
"${DIST_CMD[@]}" --master_port 28887 train_val/mmvg_train.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 4 --epochs ${STAGE_EPOCHS}  --batch_size $BATCHSIZE --lr 0.000005  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --dataset $DATA_SET      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --mixup_pretrain --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --hi_lora_retrain $OUTPUT_DIR_STAGE2/best_checkpoint.pth      --hi_lora_clip $OUTPUT_DIR_STAGE2/clip_lora_stage_with_bridge.pth       --output_dir $OUTPUT_DIR_STAGE3/     --sup_type full;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE3;
"${DIST_CMD[@]}" --master_port 28888 train_val/mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --lavs_mode ${LAVS_MODE} --lora_r_rgb ${LORA_R_RGB} --lora_r_ir ${LORA_R_IR} --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set test   --output_dir $OUTPUT_DIR_STAGE3;
