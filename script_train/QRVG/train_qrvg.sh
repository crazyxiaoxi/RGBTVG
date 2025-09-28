export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_USE_CUDA_DSA=1
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
python -m torch.distributed.launch --nproc_per_node=4 --master_port 28887 --use_env qrvg_train.py \
 --data_root ./ln_data/ \
 --batch_size 32 --lr 0.0001 --num_workers=12 \
 --output_dir ./output_training/QRNet \
 --dataset rgbtvg_flir --max_query_len 20 \
 --aug_crop --aug_scale --aug_translate \
 --lr_drop 60 --epochs 90 \
 --data_root $DATA_ROOT  --split_root $SPLIT_ROOT  --modality rgbt 