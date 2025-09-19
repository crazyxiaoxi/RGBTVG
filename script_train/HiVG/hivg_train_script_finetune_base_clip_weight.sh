export IMGSIZE=224
export BATCHSIZE=32
export MODALITY='rgbt'
export CUDADEVICES=0
bash ./script_train/HiVG/rgbtvg_flir/hivg_single_dataset_flir_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_m3fd/hivg_single_dataset_m3fd_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_mfad/hivg_single_dataset_mfad_finetune_base_clip_weight.sh
bash ./script_train/HiVG/rgbtvg_mixup/hivg_mixup_dataset_finetune_base_clip_weight.sh