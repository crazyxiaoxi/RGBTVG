#!/bin/bash
# ===================== GT Comprehensive Test Script =====================
# Automatically test GT visualization for all dataset and modality combinations
# Usage: bash run_all_gt_tests.sh

echo "Starting GT comprehensive test..."
echo "Test scope:"
echo "  - Datasets: flir, m3fd, mfad"
echo "  - Modalities: rgb, ir, rgbt"
echo "  - Total: 9 combinations"
echo "========================================"


# Define datasets and modalities
DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")
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
        
        echo ""
        echo "Test $CURRENT_TEST/$TOTAL_TESTS: $dataset + $modality"
        echo "   Output: ./visual_result/gt/$dataset/$modality"
        echo "----------------------------------------"
        
        # Run test
        if bash visualize_scripts/shell_scripts/visualize_gt.sh "$dataset" "$modality"; then
            echo "Test successful: $dataset + $modality"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "Test failed: $dataset + $modality"
            FAILED_TESTS+=("$dataset-$modality")
        fi
        
        echo "----------------------------------------"
        
        # Brief pause to avoid system overload
        sleep 1
    done
done

# End time and statistics
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "GT comprehensive test completed!"
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
echo "visual_result/gt/"
echo "├── rgbtvg_flir/"
echo "│   ├── rgb/"
echo "│   ├── ir/"
echo "│   └── rgbt/"
echo "├── rgbtvg_m3fd/"
echo "│   ├── rgb/"
echo "│   ├── ir/"
echo "│   └── rgbt/"
echo "└── rgbtvg_mfad/"
echo "    ├── rgb/"
echo "    ├── ir/"
echo "    └── rgbt/"

echo ""
echo "Tips:"
echo "   - RGBT modality generates two images: *_rgb.jpg and *_ir.jpg"
echo "   - Each test generates 100 GT visualization samples by default"
echo "   - To modify sample count, edit NUM_SAMPLES in visualize_gt.sh"
echo "   - GT file naming format: gt_000001_rgb.jpg, gt_000001_ir.jpg, gt_000001.txt"

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo ""
    echo "All GT tests completed successfully!"
    exit 0
else
    echo ""
    echo "Some GT tests failed, please check the failed list above"
    exit 1
fi
