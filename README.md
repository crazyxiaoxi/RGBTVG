

<h1>RGBT-Ground: Robust Visual Grounding in Complex Real-World Scenarios</h1>

<h2>Overview</h2>
<p>
Visual Grounding (VG) aims to localize specific objects in an image according to natural language expressions and serves as a fundamental task for vision–language understanding.
However, existing VG benchmarks are mostly derived from datasets collected under clean and controlled environments (e.g., COCO), where scene diversity is limited.
Consequently, they fail to reflect the complexity of real-world conditions such as illumination changes, adverse weather, long-distance observation, and small objects, which are critical for evaluating robustness and generalization in safety-critical applications.
</p>

<p>
To address these limitations, we present <strong>RGBT-Ground</strong>, the first large-scale visual grounding benchmark designed for complex real-world scenarios.
The benchmark consists of spatially aligned RGB and Thermal Infrared (TIR) image pairs with high-quality referring expressions, corresponding object bounding boxes,
and fine-grained annotations at the scene, environment, and object levels.
It enables comprehensive evaluation and facilitates research on robust visual grounding under diverse and challenging conditions.
</p>

<h2>RGBT-Ground Dataset</h2>
<p>
RGBT-Ground is constructed to support both uni-modal and multi-modal visual grounding research in real-world environments.
The dataset provides spatially aligned RGB–TIR image pairs, high-quality natural language referring expressions,
precise object bounding box annotations, and fine-grained annotations at scene, environment, and object levels.
It covers diverse conditions including illumination changes, weather variations, and long-distance scenarios,
making it suitable for systematic robustness evaluation beyond clean benchmark settings.
</p>

<h2>Unified Visual Grounding Framework</h2>
<p>
To support multi-sensor visual grounding research, we establish a unified visual grounding framework that supports
uni-modal inputs (RGB-only or TIR-only) as well as multi-modal inputs (RGB–TIR).
The framework enables existing single-modal visual grounding methods to be adapted to multi-modal settings
and provides consistent training and evaluation protocols for fair comparison across different modalities.
</p>

<h2>RGBT-VGNet Baseline</h2>
<p>
Based on the unified framework, we propose <strong>RGBT-VGNet</strong>, a simple yet effective baseline for multi-modal visual grounding.
RGBT-VGNet fuses complementary information from RGB and TIR modalities to achieve robust grounding performance in challenging environments.
Experimental results on RGBT-Ground show that RGBT-VGNet significantly outperforms adapted single-modal and multi-modal baselines,
particularly in nighttime and long-distance scenarios.
</p>

<h2>Repository Structure</h2>
<pre>
RGBTVG/
├── data_pre/                # Data preprocessing scripts
├── datasets/                # Dataset definitions and loaders
├── models/                  # Model architectures
├── script_train/            # Training scripts
├── script_eval/             # Evaluation scripts
├── script_visualize/        # Visualization tools
├── train_val/               # Training and validation utilities
├── utils/                   # Helper functions
├── requirements.txt         # Dependencies
└── README.md                # Documentation
</pre>

<h2>Usage</h2>
<p>
Please refer to the scripts in each directory for training, evaluation, and visualization.
Configuration files should be modified to specify dataset paths, modality settings, and experimental options.
</p>

<h2>License</h2>
<p>
All related resources, including the dataset, code, and baseline models, will be publicly released
to promote future research on robust visual grounding in complex real-world environments.
</p>

</body>
</html>
