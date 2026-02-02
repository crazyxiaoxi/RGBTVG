export IMGSIZE=224
export BATCHSIZE=32
export MODALITY='rgbt'
#export CUDADEVICES=0
export CUDADEVICES=0,1,2,3
export MODEL_NAME=MMVG #MMVG,MMVG_te

# export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint_2.pth" #modified pos embedding
#export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint_peft0112.pth" #modified peft version to 0112
# export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/${MODEL_NAME}/mixup_pretraining_base/mixup/two_encoder_best_checkpoint.pth"
export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/MMVG/mixup_pretraining_base/mixup/merged_fixed_best_checkpoint_peft0111.pth" #merged


bash ./script_train/MMVG/rgbtvg_flir/retrain_mmvg_single_dataset_flir_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_m3fd/retrain_mmvg_single_dataset_m3fd_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_mfad/retrain_mmvg_single_dataset_mfad_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_mixup/retrain_mmvg_mixup_dataset_finetune_base_clip_weight.sh
