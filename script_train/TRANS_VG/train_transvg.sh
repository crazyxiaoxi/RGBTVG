export CUDA_VISIBLE_DEVICES=3

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  
# ReferItGame
python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port 28700 transvg_train.py --batch_size 4 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Detr/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --dataset rgbtvg_flir --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --max_query_len 20 --output_dir output_training/TransVG_224_rgbt/referit_r50 --epochs 1 --lr_drop 60
