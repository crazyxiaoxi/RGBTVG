export IMGSIZE=640
export BATCHSIZE=4
export MODALITY='rgbt'
export CUDADEVICES=0
bash ./script_train/MMVG/rgbtvg_m3fd/mmvg_single_dataset_m3fd_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_flir/mmvg_single_dataset_flir_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_mfad/mmvg_single_dataset_mfad_finetune_base_clip_weight.sh
bash ./script_train/MMVG/rgbtvg_mixup/mmvg_mixup_dataset_finetune_base_clip_weight.sh