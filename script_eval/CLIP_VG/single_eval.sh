#!/bin/bash

# Dataset configuration (rgbtvg_flir / rgbtvg_m3fd / rgbtvg_mfad)
DATA_SET=${DATASET:-rgbtvg_flir}

echo -e "\n\n\n\n\n\n\n==================== clipvg single eval dataset: $DATA_SET ==========================="
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgb}            # rgb / ir / rgbt
CUDADEVICES=${CUDADEVICES:-0}

# Default evaluation split list:
#   - Global:          test
#   - illumination/*:  VWL, WL, NL, SL
#   - obj_size/*:      NS, SS
#   - occlusion/*:     PO, HO
#   - scene/*:         UB, SU, RR, HW, RS, ID, PL, IT, TN, BG, CP, MK, WF
#   - weather/*:       FY, RY, SY, CY
EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY testA testB testC val"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

# 分布式评估配置
echo "use imgsize $IMGSIZE batchsize $BATCHSIZE modality $MODALITY cudadevices $CUDADEVICES nproc_per_node $NPROC_PER_NODE"
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# 路径配置
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official/CLIP_VG_${IMGSIZE}_${MODALITY}/$DATA_SET"}

# Evaluation arguments (kept consistent with EVAL_ARGS in the training script)
EVAL_ARGS=( --num_workers 4 --modality $MODALITY --batch_size $BATCHSIZE --imsize $IMGSIZE --max_query_len 77 )

# Evaluation function
evaluate() {
    local eval_set=$1
    echo -e "\n>>>> Eval set: $eval_set, model: $EVAL_MODEL_PATH"
    "${DIST_CMD[@]}" \
        --master_port 28881 \
        train_val/eval_clip_vg.py \
        "${EVAL_ARGS[@]}" \
        --dataset "$DATA_SET" \
        --data_root "$DATA_ROOT" \
        --split_root "$SPLIT_ROOT" \
        --eval_model "$EVAL_MODEL_PATH" \
        --eval_set "$eval_set" \
        --vl_hidden_dim 512 \
        --vl_enc_layers 6 \
        --model "ViT-B/16"\
        --output_dir "$OUTPUT_DIR"
}

# Evaluation only, no training
for es in $EVAL_SETS; do
    evaluate "$es"
done
