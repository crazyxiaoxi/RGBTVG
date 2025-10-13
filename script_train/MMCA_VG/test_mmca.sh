export CUDA_VISIBLE_DEVICES=1,2,3

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled" 
# ReferItGame
python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 28887 mmca_eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset rgbtvg_flir --max_query_len 20 --eval_set test --eval_model /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/output_training/MMCA/referit_450/best_checkpoint.pth --output_dir ./outputs/referit_r50 --eval_set test --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --modality rgbt


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc.pth --output_dir ./outputs/refcoco_r50


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc+.pth --output_dir ./outputs/refcoco_plus_r50


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ../released_models/TransVG_gref.pth --output_dir ./outputs/refcocog_gsplit_r50


# # RefCOCOg u-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ../released_models/TransVG_gref_umd.pth --output_dir ./outputs/refcocog_usplit_r50