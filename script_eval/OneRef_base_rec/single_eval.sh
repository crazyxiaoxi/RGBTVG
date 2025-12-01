#!/bin/bash

# Single-dataset single-modality evaluation script for OneRef_base_rec
DATA_SET=${DATASET:-rgbtvg_flir}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-1}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-1}

EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"../dataset_and_pretrain_model/result/OneRef_B/ONEREF_base_rec_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official/ONEREF_base_rec_${IMGSIZE}_${MODALITY}/$DATA_SET"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

EVAL_ARGS=( \
  --modality $MODALITY \
  --imsize $IMGSIZE \
  --batch_size $BATCHSIZE \
  --max_query_len 64 \
  --model beit3_base_patch16_224 \
  --task grounding \
  --dataset $DATA_SET \
  --use_regress_box \
  --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
)

evaluate() {
  local eval_set=$1
  local model_path=${2:-$EVAL_MODEL_PATH}
  echo -e "\n>>>> [OneRef_base_rec] Eval set: $eval_set, model: $model_path"
  "${DIST_CMD[@]}" \
    --master_port 25000 \
    oneref_eval.py \
    "${EVAL_ARGS[@]}" \
    --finetune "$model_path" \
    --eval_set "$eval_set" \
    --eval_model "$model_path"

}

for es in $EVAL_SETS; do
  evaluate "$es" "$EVAL_MODEL_PATH"
done
