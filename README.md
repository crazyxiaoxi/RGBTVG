```markdown
# RGBT Visual Grounding Benchmark

## Introduction

Visual Grounding (VG) aims to localize specific objects in an image according to natural language expressions and serves as a fundamental task for vision–language understanding. Existing VG benchmarks are mainly derived from datasets collected under clean and controlled environments (e.g., COCO), where scene diversity is limited and objects are visually salient. As a result, these benchmarks fail to adequately reflect the complexity of real-world scenarios, such as illumination changes, adverse weather conditions, long-distance observation, and small or low-contrast objects. This limitation restricts the evaluation of model robustness and generalization, especially for safety-critical applications.

To address these challenges, we introduce **RGBT-Ground**, the first large-scale visual grounding benchmark specifically designed for complex real-world environments. The dataset consists of spatially aligned RGB and Thermal Infrared (TIR) image pairs, along with high-quality natural language referring expressions, corresponding object bounding boxes, and fine-grained annotations at the scene, environment, and object levels. This benchmark enables comprehensive evaluation and facilitates research on robust visual grounding under all-day and all-weather conditions.

---

## RGBT-Ground Dataset

RGBT-Ground is constructed to support both uni-modal and multi-modal visual grounding research in challenging real-world scenarios. The dataset provides:

- Spatially aligned RGB–TIR image pairs  
- High-quality referring expressions for each target object  
- Precise bounding box annotations  
- Fine-grained annotations at scene, environment, and object levels  
- Coverage of diverse conditions, including varying illumination, weather, and viewing distances  

These properties make RGBT-Ground suitable for evaluating robustness and generalization of visual grounding models beyond clean benchmark settings.

---

## Unified Visual Grounding Framework

To enable fair and flexible evaluation, we establish a unified visual grounding framework that supports:

- Uni-modal visual inputs (RGB-only or TIR-only)  
- Multi-modal visual inputs (RGB–TIR)  
- Adaptation of existing visual grounding methods to multi-modal settings  
- Consistent training, evaluation, and comparison across different modalities  

This framework allows existing VG models, originally designed for single-sensor inputs, to be extended and evaluated under multi-modal conditions.

---

## RGBT-VGNet Baseline

Based on the unified framework, we propose **RGBT-VGNet**, a simple yet effective baseline model for multi-modal visual grounding. RGBT-VGNet focuses on fusing complementary information from RGB and TIR modalities to achieve robust grounding performance in challenging environments.

Experimental results on RGBT-Ground demonstrate that RGBT-VGNet significantly outperforms adapted single-modal and multi-modal baselines, particularly in nighttime and long-distance scenarios, highlighting the advantages of multi-sensor fusion for robust visual grounding.

---

## Repository Structure

```

RGBTVG/
├── data_pre/                # Data preprocessing scripts
├── datasets/                # Dataset definitions and loaders
├── models/                  # Model architectures
├── script_train/            # Training scripts
├── script_eval/             # Evaluation scripts
├── script_visualize/        # Visualization tools
├── train_val/               # Training and validation utilities
├── utils/                   # Helper functions and common utilities
├── requirements.txt         # Dependency list
└── README.md                # Project documentation

```

---

## Usage

Please refer to the scripts in the corresponding directories for training, evaluation, and visualization. Configuration files should be adjusted to specify dataset paths, modality settings, and experimental options.

---

## License and Availability

All related resources, including the dataset, code, and baseline models, will be publicly released to facilitate future research on robust visual grounding in complex real-world environments.

```
