<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RGBT-Ground: Robust Visual Grounding in Complex Real-World Scenarios</title>
</head>
<body>

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

<h2>Downloads</h2>
<ul>
    <li><strong>Dataset</strong>: Download the RGBT-Ground dataset from Hugging Face:<br>
        <a href="https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset">https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset</a>
    </li>
    <li><strong>Pre-trained Models</strong>: Download the baseline models (RGBT-VGNet) from Hugging Face:<br>
        <a href="https://huggingface.co/JiawenXi/RGBT-Ground-Model">https://huggingface.co/JiawenXi/RGBT-Ground-Model</a>
    </li>
</ul>

<h2>Installation and Usage</h2>

<h3>Environment Setup</h3>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/crazyxiaoxi/RGBTVG.git
cd RGBTVG</code></pre>
    </li>
    <li>Create and activate the conda environment:
        <pre><code>conda env create -f environment_full.yml
conda activate rgbtvg  # Replace with your desired environment name if needed</code></pre>
        <p><strong>Note</strong>: The <code>environment_full.yml</code> file includes all required dependencies, but may cause conflicts.Install according to the actual situation of the local environment. </p>
    </li>
</ol>

<h3>Dataset Preparation</h3>
<p>Download the dataset using the links above and place it in the appropriate directory (e.g., specify the path in the configuration files under <code>../dataset_and_pretrain_model/datasets/VG</code> or as required by the scripts).</p>

<h3>Training</h3>
<p>Run the training scripts for the desired model:</p>
<pre><code>bash script_train/&lt;model_name&gt;/all.sh</code></pre>
<p>Modify the corresponding configuration files (e.g., for dataset paths, modality settings, hyperparameters) before running.</p>

<h3>Evaluation / Inference</h3>
<p>Run evaluation on trained models:</p>
<pre><code>bash script_eval/run_all_evals.sh</code></pre>
<p>This script will perform inference and compute metrics across different settings.</p>

<h3>Visualization</h3>
<p>Visualize grounding results:</p>
<pre><code>bash script_visualize/all/run_all_&lt;name&gt;_tests.sh</code></pre>
<p>Replace <code>&lt;name&gt;</code> with the specific test split or configuration as needed.</p>

<h2>Repository Structure</h2>
<pre>
RGBTVG/
├── data_pre/              # Data preprocessing scripts
├── datasets/              # Dataset definitions and loaders
├── models/                # Model architectures
├── script_train/          # Training scripts
├── script_eval/           # Evaluation scripts
├── script_visualize/      # Visualization tools
├── train_val/             # Training and validation utilities
├── utils/                 # Helper functions
├── environment_full.yml   # Full conda environment specification
├── requirements.txt       # Dependencies (alternative to yml)
└── README.md              # Documentation
</pre>

<h2>License</h2>
<p>
All related resources, including the dataset, code, and baseline models, are publicly released to promote future research on robust visual grounding in complex real-world environments.
</p>

</body>
</html>
