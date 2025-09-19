export IMGSIZE=224
export BATCHSIZE=32
export MODALITY='rgb'
export CUDADEVICES=0
# export CUDADEVICES=0,1,2,3,4,5,6,7
# export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/best_checkpoint.pth"
export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint.pth"
# export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint_2.pth" #modified pos embedding
bash ./script_train/HiVG/rgbtvg_flir/retrain_hivg_single_dataset_flir_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_m3fd/retrain_hivg_single_dataset_m3fd_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_mfad/retrain_hivg_single_dataset_mfad_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_mixup/retrain_hivg_mixup_dataset_finetune_base_clip_weight.sh
