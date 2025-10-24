export CUDA_VISIBLE_DEVICES=3
# refcocog-g
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  

python -m torch.distributed.launch --master_port 28600 --nproc_per_node=1 --use_env mdetr_eval.py --model_type ResNet --batch_size 16 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --eval_model /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/output_training/MDETR_rgbt/best_checkpoint.pth --eval_set testA --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --model_name MDETR --modality rgbt  --dataset rgbtvg_flir 