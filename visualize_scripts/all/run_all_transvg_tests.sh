#!/bin/bash
# ===================== TransVG Comprehensive Test Script =====================
# Automatically test all dataset and modality combinations
# Usage: bash run_all_transvg_tests.sh

echo "Starting TransVG comprehensive test..."
echo "Test scope:"
echo "  - Datasets: flir, m3fd, mfad"
echo "  - Modalities: rgb, ir, rgbt"
echo "  - Total: 9 combinations"
echo "========================================"


# Define datasets and modalities
DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")

# Model base path
MODEL_BASE_PATH="../dataset_and_pretrain_model/result/transvg"

# Counter
TOTAL_TESTS=9
CURRENT_TEST=0
SUCCESS_COUNT=0
FAILED_TESTS=()

# Start time
START_TIME=$(date +%s)

# Iterate through all combinations
for dataset in "${DATASETS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        # Build model path
        # Extract short name from dataset name (rgbtvg_flir -> flir)
        DATASET_SHORT=$(echo $dataset | sed 's/rgbtvg_//')
        MODEL_CHECKPOINT="${MODEL_BASE_PATH}/TransVG_224_${modality}_${DATASET_SHORT}_best.pth"
        
        echo ""
        echo "Test $CURRENT_TEST/$TOTAL_TESTS: $dataset + $modality"
        echo "   Model: $MODEL_CHECKPOINT"
        echo "   Output: ./visual_result/transvg/$dataset/$modality"
        echo "----------------------------------------"
        
        # Check if model file exists
        if [ ! -f "$MODEL_CHECKPOINT" ]; then
            echo "Warning: Model file does not exist: $MODEL_CHECKPOINT"
            echo "   Skipping this test..."
            FAILED_TESTS+=("$dataset-$modality (model file not found)")
            continue
        fi
        
        # Run test
        if bash visualize_scripts/shell_scripts/visualize_transvg.sh "$dataset" "$modality" "$MODEL_CHECKPOINT"; then
            echo "Test successful: $dataset + $modality"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "Test failed: $dataset + $modality"
            FAILED_TESTS+=("$dataset-$modality")
        fi
        
        echo "----------------------------------------"
        
        # Brief pause to avoid GPU overload
        sleep 2
    done
done

# End time and statistics
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "TransVG comprehensive test completed!"
echo "========================================"
echo "Test statistics:"
echo "   Total tests: $TOTAL_TESTS"
echo "   Success: $SUCCESS_COUNT"
echo "   Failed: $((TOTAL_TESTS - SUCCESS_COUNT))"
echo "   Duration: ${MINUTES}m ${SECONDS}s"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for failed in "${FAILED_TESTS[@]}"; do
        echo "   - $failed"
    done
fi

echo ""
echo "Result directory structure:"
echo "visual_result/transvg/"
echo "├── rgbtvg_flir/"
echo "│   ├── rgb/"
echo "│   └── ir/"
echo "├── rgbtvg_m3fd/"
echo "│   ├── rgb/"
echo "│   └── ir/"
echo "└── rgbtvg_mfad/"
echo "    ├── rgb/"
echo "    └── ir/"

echo ""
echo "Note:"
echo "   - TransVG currently lacks RGBT modality model files"
echo "   - If RGBT testing is needed, please train the corresponding model first"
echo "   - Each test generates 100 samples by default"
echo "   - To modify sample count, edit NUM_SAMPLES in visualize_transvg.sh"

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo ""
    echo "All tests completed successfully!"
    exit 0
else
    echo ""
    echo "Some tests failed, please check the failed list above"
    exit 1
fi
