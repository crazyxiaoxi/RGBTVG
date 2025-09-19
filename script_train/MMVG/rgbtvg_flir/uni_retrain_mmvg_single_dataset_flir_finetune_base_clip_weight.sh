# flir train from clip weight
DATA_SET="rgbtvg_flir"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
MODEL_NAME=${MODEL_NAME:MMVG}
RETRAIN=${RETRAIN:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint_2.pth"}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
CLIP_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_b_ml_cascade_maskrcnn_model_224.pth"  
OUTPUT_DIR="./output_retraining/${MODEL_NAME}_${IMGSIZE}_${MODALITY}/$DATA_SET/rgbt_finetuning_base_clip_weight/output"

echo -e "\n\n\n\n\n\n\n==================== flir warmup ==========================="
"${DIST_CMD[@]}" --master_port 28887 mmvg_train.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 4 --epochs 120  --batch_size $BATCHSIZE --lr 0.00025   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512  --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --dataset $DATA_SET      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss   --data_root $DATA_ROOT  --split_root $SPLIT_ROOT   --output_dir $OUTPUT_DIR  --retrain $RETRAIN;
"${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR/best_checkpoint.pth --eval_set val      --output_dir $OUTPUT_DIR;
"${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR/best_checkpoint.pth --eval_set test     --output_dir $OUTPUT_DIR;
"${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR/best_checkpoint.pth --eval_set testA    --output_dir $OUTPUT_DIR;
"${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR/best_checkpoint.pth --eval_set testB    --output_dir $OUTPUT_DIR;
"${DIST_CMD[@]}" --master_port 28888 mmvg_eval.py --model_name $MODEL_NAME --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATA_SET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUTPUT_DIR/best_checkpoint.pth --eval_set testC    --output_dir $OUTPUT_DIR;
##
