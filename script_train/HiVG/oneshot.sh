# flir train from clip weight
DATASET=${DATASET:-flir}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
RETRAIN=${RETRAIN:-""}
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
CLIP_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/CLIP/clip_l_ml_cascade_maskrcnn_model_224.pth"  
OUTPUT_DIR="./oneshot_logs/hivg"
"${DIST_CMD[@]}" --master_port 28888 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATASET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss   --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $RETRAIN     --eval_set val    --output_dir $OUTPUT_DIR  --vl_hidden_dim 512 ;
"${DIST_CMD[@]}" --master_port 28888 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATASET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss   --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $RETRAIN      --eval_set test   --output_dir $OUTPUT_DIR --vl_hidden_dim 512 ;
"${DIST_CMD[@]}" --master_port 28888 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATASET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss   --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $RETRAIN      --eval_set testA  --output_dir $OUTPUT_DIR --vl_hidden_dim 512 ;
"${DIST_CMD[@]}" --master_port 28888 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATASET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss   --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $RETRAIN      --eval_set testB  --output_dir $OUTPUT_DIR --vl_hidden_dim 512 ;
"${DIST_CMD[@]}" --master_port 28888 hivg_eval.py --modality $MODALITY --num_workers 2 --batch_size $BATCHSIZE  --dataset $DATASET            --imsize $IMGSIZE --max_query_len 77 --normalize_before --mixup_pretrain    --use_mask_loss   --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $RETRAIN      --eval_set testC  --output_dir $OUTPUT_DIR --vl_hidden_dim 512 ;