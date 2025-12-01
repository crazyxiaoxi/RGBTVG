#!/bin/bash

# Run all evaluation scripts (all_eval and oneshot_eval) 

# All-evaluation scripts (3 datasets × 3 modalities where applicable)
bash ./script_eval/CLIP_VG/all_eval.sh
bash ./script_eval/HiVG/all_eval.sh
bash ./script_eval/HiVG_L/all_eval.sh
bash ./script_eval/MDETR/all_eval.sh
bash ./script_eval/MDETR_clip/all_eval.sh
bash ./script_eval/MMCA_VG/all_eval.sh
bash ./script_eval/MMVG/all_eval.sh
bash ./script_eval/OneRef_base_rec/all_eval.sh
bash ./script_eval/OneRef_large_rec/all_eval.sh
bash ./script_eval/TRANS_VG/all_eval.sh

# Oneshot evaluation scripts (3 datasets × 2 modalities rgb/ir where applicable)
bash ./script_eval/CLIP_VG/oneshot_eval.sh
bash ./script_eval/HiVG/oneshot_eval.sh
bash ./script_eval/HiVG_L/oneshot_eval.sh
bash ./script_eval/OneRef_base_rec/oneshot_eval.sh
bash ./script_eval/OneRef_large_rec/oneshot_eval.sh
