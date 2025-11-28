DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")


MODEL_BASE_PATH="../dataset_and_pretrain_model/result/MDETR_clip"
TOTAL_TESTS=9
CURRENT_TEST=0
SUCCESS_COUNT=0
FAILED_TESTS=()

START_TIME=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        DATASET_SHORT=$(echo $dataset | sed 's/rgbtvg_//')
        MODEL_CHECKPOINT="${MODEL_BASE_PATH}/MDETR_224_clip_${modality}_${DATASET_SHORT}_best.pth"

        if [ ! -f "$MODEL_CHECKPOINT" ]; then
            FAILED_TESTS+=("$dataset-$modality not exist")
            continue
        fi
        
        # 运行测试
        if bash visualize_scripts/shell_scripts/visualize_mdetr_clip.sh "$dataset" "$modality" "$MODEL_CHECKPOINT"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAILED_TESTS+=("$dataset-$modality")
        fi
        
        echo "----------------------------------------"
        
        sleep 2
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

