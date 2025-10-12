export CUDA_VISIBLE_DEVICES=3
export TORCH_USE_CUDA_DSA=1
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"  
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"  


python -m torch.distributed.launch --nproc_per_node=1 --use_env transvg_eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset rgbtvg_flir --max_query_len 20 --eval_set test --eval_model /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/output_training/TransVG_224_rgbt/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50  --eval_set test --data_root $DATA_ROOT  --split_root $SPLIT_ROOT

