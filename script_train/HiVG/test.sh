# flir train from clip weight

DATA_SET=${DATASET:-rgbtvg_flir}
echo -e "\n\n\n\n\n\n\n==================== mdetr single dataset: $DATA_SET ==========================="
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
CLIP_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224_peft0111_nolora.pth"  
OUTPUT_DIR_WARMUP="./output_training/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v100"
OUTPUT_DIR_STAGE1="./output_training/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v101"
OUTPUT_DIR_STAGE2="./output_training/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v102"
OUTPUT_DIR_STAGE3="./output_training/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output_v103"


# stage 3

"${DIST_CMD[@]}" --master_port 28873 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET           --vl_hidden_dim 512 --imsize $IMGSIZE --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR_STAGE3/best_checkpoint.pth      --eval_set val    --output_dir $OUTPUT_DIR_STAGE3;
