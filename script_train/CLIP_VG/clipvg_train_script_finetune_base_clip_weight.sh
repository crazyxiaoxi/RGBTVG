export IMGSIZE=640
export BATCHSIZE=4
MODALITY='rgb'
export CUDADEVICES=0
bash ./script_train/CLIP_VG/rgbt_flir/clipvg_single_dataset_flir_train_and_eval_full_sup.sh
bash ./script_train/CLIP_VG/rgbt_m3fd/clipvg_single_dataset_m3fd_train_and_eval_full_sup.sh
bash ./script_train/CLIP_VG/rgbt_mfad/clipvg_single_dataset_mfad_train_and_eval_full_sup.sh
bash ./script_train/CLIP_VG/rgbt_mixup/clipvg_mixup_dataset_train_and_eval_full_sup.sh