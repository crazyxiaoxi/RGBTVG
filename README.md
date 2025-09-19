<p align="center"> <h1 align="center">RGBT Visual Grounding Benchmark</h1>
  <p align="center">
    <b> UnderReview </b>
    <!-- <br />
    <a href="https://scholar.google.com.hk/citations?user=4rTE4ogAAAAJ&hl=zh-CN&oi=sra"><strong> Linhui Xiao </strong></a>
    ·
    <a href="https://yangxs.ac.cn/home"><strong>Xiaoshan Yang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=HBZ9plsAAAAJ&hl=zh-CN"><strong>Fang Peng </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN"><strong>Yaowei Wang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=hI9NRDkAAAAJ&hl=zh-CN"><strong>Changsheng Xu</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/pdf/2404.13400'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://openreview.net/forum?id=NMMyGy1kKZ'>
      <img src='https://img.shields.io/badge/ACM MM 2024-purple' alt='arXiv PDF'>
    </a>
    <a href='docs/ACM_MM_2024_HiVG_poster.pdf'>
      <img src='https://img.shields.io/badge/ACM MM Poster-lightblue' alt='arXiv PDF'>
    </a>
<br />


<p align="center"> <img src='docs/model.jpg' align="center" width="55%"> </p> -->

This repository provides the first large-scale RGB-Thermal (RGBT) Visual Grounding Benchmark, constructed from autonomous driving scenarios. Our benchmark includes:

* **RGBTVG-40K Dataset:** 40000 high-quality images (20000 RGB-Thermal image pairs) with referring expressions and bounding box annotations.
* **A Flexible Visual Grounding Toolbox:** Supports both single-modal (RGB or thermal) and multi-modal (RGB+thermal) visual grounding experiments.
* **Extensive Baseline and Comparison Experiments:** Includes scripts and results for a series of state-of-the-art visual grounding models, facilitating fair and reproducible evaluation.

<!-- This repository is the official Pytorch implementation for the paper [**RGBT Visual Grounding Benchmark**](https://arxiv.org/abs/2404.13400), which is based on the **CLIP-VG** ([github](https://github.com/linhuixiao/CLIP-VG), [publication](https://ieeexplore.ieee.org/abstract/document/10269126), [Arxiv](https://arxiv.org/abs/2305.08685)).  -->

If you have any questions, please feel free to open an issue or contact me with emails: <ty_zhao@buaa.edu.cn>.

<h3 align="left">
Links: 
<a href="https://arxiv.org/pdf/2404.13400">ArXiv</a>, 
<a href="https://openreview.net/forum?id=NMMyGy1kKZ">Paper</a>
</h3>

**Please leave a <font color='orange'>STAR ⭐</font> if you like this project!**

-----------------


## TO XJW
* copy dataset and pretrain model
* python data_pre/clip_peft_0111.py
* folder: models
* folder: script_eval
* folder: script_train
* file: xxx_train.py
* file: xxx_eval.py
* file: engine.py

## RGBTVG-40K Dataset

 ### Various Training Settings
 * **Different Modality:** RGBT (RGB-IR), RGB, IR
 * **Different Image Size:** 224, 640, 1280
 * **Different Training setting:** Mixup-Dataset, Single dataset


 ### **RGBTVG-40K** Dataset Overall Information
|Dataset| Total Images|Total Instance|Total Querys|Train  |Val    |Test   |TestA  |TestB  |TestC  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|ALL (Mixup)|43072 (21536 pairs)|39602 |39602|27000  |2045   |10557  |2858   |4306   |5779   |
| FLIR  |   10284 (5142 pairs)  |9738  |9738 | 7000  | 608   |2130   |837    |640    |986    |
| M3FD  |   8400 (4200 pairs)   |8358  |8358 | 4000  |181    |4177   |1232   |1214   |2243   |
| MFAD  |   24388 (12194 pairs) |21506 |21506| 16000 |1256   |4250   |789    |2452   |2550   |

 ### **RGBTVG-40K** Label's Lightning Distribution
|Dataset|   0   |   1   |   2   |   3   |   4   |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ALL (Mixup)|0|318|16651|15826|6807|
|FLIR|0|49|1962|1387|6340|
|M3FD|0|128|2320|5491|419|
|MFAD|0|141|12369|8948|48|




 ### **RGBTVG-40K** Label's Small Object Distribution
|Dataset| Total | train |  val  | test  | testA | testB | testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|ALL (Mixup)|22831|15812|1240   |5779   |0      |2386   |5779   |
|FLIR       |5493 |4157 |350    |986    |0      |333    |986    | 
|M3FD       |4564 |2210 |111    |2243   |0      |512    |2243   |
|MFAD       |12774|9445 |779    |2550   |0      |1541   |2550   |

### The Prompt of Ligntning Label Generation
```python
"text": f'''You are an expert at analyzing lighting conditions in images. I will provide an image, and your task is to classify the overall lighting intensity of the image into one of the following categories:
    0. No Light: The image appears almost completely black.
    1. Very Weak Light: The image is mostly dark, but some features can be faintly seen.
    2. Weak Light: The image has low visibility, typical for dawn, dusk or cloudy day
    3. Normal Light: The image has normal daylight brightness
    4. Strong Light: The image is brightly lit, typically from midday sunlight, or there may be a bright light source
    
    Please return only one number corresponding to the lighting condition: 0 (no_light), 1 (very_weak_light), 2 (weak_light), 3 (normal_light), or 4 (strong_light).
    
    Now, please begin generating the number'''
```
### The Prompt of Ligntning Label Generation

```python
    "text": f'''I will provide an image and the bounding box coordinates (bbox) of a {category_name}. Your task is to describe the object within the bounding box in one concise sentence, focusing on its appearance and key features.
    
    1. Object Details: Please describe the object in detail, including but not limited to its Position in the picture, appearance, color, shape, texture, size, and posture.  
    2. Contextual Relationship: If there is any relationship between the objects within the image or between the object and the background (e.g., relative positioning, interaction), please reflect this in your description.  
    3. Distinguishing Similar Objects: If there are multiple similar objects in the image, differentiate them by comparing their details such as color intensity, position, state, etc.  
    4. Concise and Clear Language: Provide a single, concise sentence that captures the key features without breaking down into different aspects. The description should be rich in information yet simple for later data processing.

    Please generate only one few words of description for the {category_name} with the following bounding box coordinates:[{bbox}]
    
    Now, please begin generating the description:'''
```


## Supported Methods (Plan)

|Methods| Venue | Links | Already Supported|
|:-----:|:-----:|:-----:|:-----:|
FAOA |ICCV'19|https://github.com/zyang-ur/onestage_grounding |
ReSC | ECCV'20 |https://github.com/zyang-ur/ReSC|
TransVG | ICCV'21| https://github.com/djiajunustc/TransVG.git|
QRNet | CVPR'22 |https://github.com/LukeForeverYoung/QRNet.git|
CLIP-VG |TMM'23| https://github.com/linhuixiao/CLIP-VG.git |ING
HiVG| ACMMM'24|https://github.com/linhuixiao/HiVG.git |YES
MMCA |ACMMM'24|https://github.com/Mr-Bigworth/MMCA |
D-MDETR|TPAMI'24| https://github.com/MCG-NJU/Dynamic-MDETR |
OneRef    | NeurIPS'24 | https://github.com/linhuixiao/OneRef.git |

## Methods
### Asymmetry-Dual Lora  
### Text-Guided Hierarchical Multimodal Fusion

## Experiments

## TODO: MMVG hi-du-lora(16,48)探究是否是四阶段加载所带来的问题，将网络合成统一的框架，不需要多次加载 0; 
### Training Dataset: Mixup RefC, ReferIt, Flickr, ImageSize: 224, Modality: RGB
|Methods| Venue |Visual/Language Backbone |val|testA|testB|val|testA|testB|val|test|test|test|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
HiVG-B (paper) | ACMMM'24 | CLIP-B/CLIP-B |90.56|92.55|87.23|83.08|89.21|76.68|84.52|85.62|77.75|82.08
HiVG-B (ours) | ACMMM'24 | CLIP-B/CLIP-B |88.63|90.35|84.37|79.32|84.29|71.61|82.20|81.63|76.61(val)-74.97(test)|79.60(val)-80.83(test)

### Ours. ImageSize: 224, Modality: RGBT，Pretrained on RefC, ReferIt, Flickr.
|Methods| Hyperparam |Visual/Language Backbone |  val  | test  | testA | testB | testC | val  | test  | testA | testB | testC | val  | test  | testA | testB | testC | 校验forward|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
||||rgbtvg-flir|||||rgbtvg-m3fd|||||rgbtvg-mfad|||||
Ours (CLIP freeze) | - | CLIP-B/CLIP-B ||||||
Ours (Clip-Adapter,0) | - | CLIP-B/CLIP-B ||||||
Ours (Clip-Adapter,-1) | - | CLIP-B/CLIP-B ||||||
Ours (Clip-Adapter,hi) | - | CLIP-B/CLIP-B ||||||
Ours (Two-Clip-Encoder w./ Hi-Lora) | nn.linear fusoin | CLIP-B/CLIP-B todo0|72.03|71.25|88.33|68.43|52.52|65.76|66.34|95.12|71.21|42.79|67.64|65.90|90.40|63.23|49.09|yes
Ours (Two-Clip-Encoder w./ Hi-Lora) mixup| nn.linear fusoin | CLIP-B/CLIP-B todo0|73.51|72.98|89.28|70.93|54.73|66.30|68.83|95.04|74.67|46.84|67.59|66.23|88.51|64.45|50.07|yes
Ours (Two-Clip-Encoder w./ Hi-Lora) | cross-former fusoin | CLIP-B/CLIP-B todo1|58.22|56.41|75.59|59.06|32.86|50.00|54.03|83.19|61.01|28.73|58.43|58.22|82.95|56.51|40.51|yes
Ours (Two-Clip-Encoder w./ Hi-Lora) mixup| cross-former fusoin | CLIP-B/CLIP-B todo1|67.10|60.72|79.88|63.43|36.59|59.78|63.57|90.34|69.07|40.74|60.74|60.73|86.36|58.79|42.98|yes
Ours (Two-ClipEncoder+Cross-Modality Trans.) | - | CLIP-B/CLIP-B ||||||
Ours (SFT-ALL) | - | CLIP-B/CLIP-B ||||||
Ours (Hi-SFT-ALL) | - | CLIP-B/CLIP-B |||||||
Ours (Hi-SFT-ALL) mixup| - | CLIP-B/CLIP-B |||||||
Ours (Hi-SFT-Part) | - | CLIP-B/CLIP-B ||||||
(Hi-Lora) | - | CLIP-B/CLIP-B|||||||
(Hi-Lora) mixup | - | CLIP-B/CLIP-B ||||||
Ours (Hi-Adalora) | - | CLIP-B/CLIP-B ||||||
Ours (Hi-SFT w./ Hi-Lora) | - | CLIP-B/CLIP-B ||||||
Ours (Hi-SFT w./ Hi-Adalora) | - | CLIP-B/CLIP-B ||||||


### Training Dataset: Single RGBTVG_RefFLIR, ImageSize: 224, Modality: RGBT，Pretrained on RefC, ReferIt, Flickr.
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |校验forward|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |70.72|73.22|92.50|67.34|52.21|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |70.72|73.22|92.50|67.34|52.21|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |72.03|73.78|93.57|67.34|52.72|
HiVG-B w/2.ml_visu_proj | ACMMM'24 | CLIP-B/CLIP-B |72.03|73.78|93.57|67.34|52.72|
HiVG-B  | ACMMM'24 | CLIP-B/CLIP-B |75.49|72.79|91.07|67.34|52.21|
MMVG-B-Lora(Hi-Dual-lora) (Warm up)| ACMMM'24 | CLIP-B/CLIP-B |72.36|72.61|89.52|68.43|54.33|
MMVG-B-Lora(Hi-Dual-lora) lora-ir adapter for text| ACMMM'24 | CLIP-B/CLIP-B |72.69|72.61|89.88|68.75|53.83|
MMVG-B-Lora(Hi-Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |74.83|73.12|89.88|69.53|54.43|
MMVG-B-Lora(Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |75.16|72.33|90.47|67.81|52.31
MMVG-B-Lora(lora-ir-only) no-lora for text| ACMMM'24 | CLIP-B/CLIP-B |74.67|72.98|91.90|68.12|52.62|
MMVG-B-Lora(As-Hi-Dual-lora_8_32) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |74.34|70.64|90.23|63.43|49.09|
MMVG-B-Lora(As-Hi-Dual-lora_16_64) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |74.50|72.70|90.00|69.37|52.88|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint| ACMMM'24 | CLIP-B/CLIP-B |72.53|72.98|90.59|69.06|53.02|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |74.50|72.14|88.69|68.59|53.93|
MMVG-B-Lora(As-Hi-Dual-lora)(16,48) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |74.17|71.76|89.64|67.96|51.81|
Ours  | - | CLIP-B/CLIP-B ||||||


### Training Dataset: Single RGBTVG_RefFLIR, ImageSize: 224, Modality: RGB，Pretrained on RefC, ReferIt, Flickr.
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |校验forward|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |67.10|68.63|90.11|57.18|46.37|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |68.58|68.63|90.59|55.15|45.96|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |68.58|68.63|90.59|55.15|45.96|
HiVG-B | ACMMM'24 | CLIP-B/CLIP-B |68.58|68.63|90.59|55.15|45.96|
Ours  | - | CLIP-B/CLIP-B ||||||


### Training Dataset: Single RGBTVG_RefM$^3$FD, ImageSize: 224, Modality: RGBT, Pretrained on RefC, ReferIt, Flickr.
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |66.84|66.51|95.37|70.39|42.92|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |66.84|68.06|95.04|73.43|45.64|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |66.84|68.06|95.04|73.43|45.64|
HiVG-B w/2.ml_visu_proj | ACMMM'24 | CLIP-B/CLIP-B |67.83|69.61|95.37|73.93|46.70|
HiVG-B  | ACMMM'24 | CLIP-B/CLIP-B |71.19|69.04|95.21|75.65|46.97
MMVG-B-Lora(Hi-Dual-lora) (Warm up)| ACMMM'24 | CLIP-B/CLIP-B |65.21|68.35|94.56|74.83|46.21|
MMVG-B-Lora(Hi-Dual-lora) lora-ir adapter for text| ACMMM'24 | CLIP-B/CLIP-B |67.93|69.47|94.80|76.06|48.13|
MMVG-B-Lora(Hi-Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |69.56|68.69|95.37|74.17|46.97|
MMVG-B-Lora(Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |66.30|67.97|93.91|73.51|46.30|
MMVG-B-Lora(lora-ir-only) no-lora for text| ACMMM'24 | CLIP-B/CLIP-B |68.47|68.95|94.56|75.08|47.46|
MMVG-B-Lora(As-Hi-Dual-lora_8_32) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |68.47|68.90|95.45|75.41|46.97|
MMVG-B-Lora(As-Hi-Dual-lora_16_64) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |65.76|67.90|94.72|73.43|45.64|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint| ACMMM'24 | CLIP-B/CLIP-B |67.93|68.45|95.21|73.35|46.79|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |71.19|69.55|95.21|74.91|48.30|
MMVG-B-Lora(As-Hi-Dual-lora)(16,48) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |67.93|69.59|94.88|74.34|48.75|
Ours  | - | CLIP-B/CLIP-B ||||||

### Training Dataset: Single RGBTVG_RefM$^3$FD, ImageSize: 224, Modality: RGB, Pretrained on RefC, ReferIt, Flickr.
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |61.95|62.71|94.23|65.04|37.36|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |63.04|64.34|94.80|66.85|39.54|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |63.04|64.34|94.80|66.85|39.54|
HiVG-B | ACMMM'24 | CLIP-B/CLIP-B |63.04|64.34|94.80|66.85|39.54|
Ours  | - | CLIP-B/CLIP-B ||||||

### Training Dataset: Single RGBTVG_RefMFAD, ImageSize: 224, Modality: RGBT
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |66.95|65.71|91.28|62.86|48.23|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |67.91|66.47|90.65|63.80|49.64|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |68.07|67.36|91.79|64.25|50.74|
HiVG-B w/2.ml_visu_proj | ACMMM'24 | CLIP-B/CLIP-B |68.55|66.84|90.78|64.00|50.58|
HiVG-B  | ACMMM'24 | CLIP-B/CLIP-B |69.10|67.34|90.65|64.65|51.17|
MMVG-B-Lora(Hi-Dual-lora) (Warm up)| ACMMM'24 | CLIP-B/CLIP-B |66.79|66.02|91.03|63.11|49.17|
MMVG-B-Lora(Hi-Dual-lora) lora-ir adapter for text| ACMMM'24 | CLIP-B/CLIP-B |68.39|67.76|91.16|64.53|51.68|
MMVG-B-Lora(Hi-Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |68.31|67.52|90.15|65.26|51.95|
MMVG-B-Lora(Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |67.99|66.79|90.78|64.12|50.35|
MMVG-B-Lora(lora-ir-only) no-lora for text| ACMMM'24 | CLIP-B/CLIP-B |68.63|67.83|90.78|65.18|51.99|
MMVG-B-Lora(As-Hi-Dual-lora_8_32) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |68.78|66.61|90.78|64.33|50.07|
MMVG-B-Lora(As-Hi-Dual-lora_16_64) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |69.18|66.61|89.77|63.92|50.35|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint| ACMMM'24 | CLIP-B/CLIP-B |68.78|66.51|89.64|64.08|50.03|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |69.58|67.57|90.78|64.94|51.68|
MMVG-B-Lora(As-Hi-Dual-lora)(16,48) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |69.42|66.82|89.52|64.04|50.70|
Ours  | - | CLIP-B/CLIP-B ||||||

### Training Dataset: Single RGBTVG_RefMFAD, ImageSize: 224, Modality: RGB
|Methods| Venue |Visual/Language Backbone |  val  | test  | testA | testB | testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warm up) | ACMMM'24 | CLIP-B/CLIP-B |64.25|62.87|90.78|59.20|44.47|
HiVG-B (Stage-1) | ACMMM'24 | CLIP-B/CLIP-B |65.20|63.65|91.79|59.89|45.41|
HiVG-B (Stage-2) | ACMMM'24 | CLIP-B/CLIP-B |66.00|63.76|90.78|60.50|45.96|
HiVG-B | ACMMM'24 | CLIP-B/CLIP-B |66.00|63.76|90.78|60.50|45.96|
Ours  | - | CLIP-B/CLIP-B ||||||

### Training Dataset: Single RGBTVG_Mixup, ImageSize: 224, Modality: RGBT
* val and test datasets orders：Mixup, RefFLIR, RefM$^3$FD, RefMFAD
  
|Methods| Venue |Visual/Language Backbone |flir_val|flir_test|flir_testA|flir_testB|flir_testC|m3fd_val|m3fd_test|m3fd_testA| m3fd_testB | m3fd_testC |mfad_val  | mfad_test  | mfad_testA | mfad_testB | mfad_testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warmup) | ACMMM'24 | CLIP-B/CLIP-B |71.85|73.59|90.23|71.25|54.43|67.93|68.64|95.37|76.15|46.57|68.15|65.90|90.65|63.39|48.90|
HiVG-B w/2.ml_visu_proj | ACMMM'24 | CLIP-B/CLIP-B |74.34|75.00|90.71|73.59|56.85|66.84|70.00|95.29|77.71|48.75|69.74|67.69|90.65|65.47|51.44|
HiVG-B  | ACMMM'24 | CLIP-B/CLIP-B |74.01|73.17|89.40|69.68|54.93|69.02|70.45|95.29|76.72|49.73|69.18|67.76|90.65|65.10|51.72|
MMVG-B-Lora(Hi-Dual-lora) (Warm up)| ACMMM'24 | CLIP-B/CLIP-B |71.71|74.20|91.54|71.25|54.83|66.84|69.23|95.53|75.74|47.19|66.16|65.60|90.78|63.68|48.35|
MMVG-B-Lora(Hi-Dual-lora) lora-ir adapter for text | ACMMM'24 | CLIP-B/CLIP-B |73.35|73.64|90.83|69.21|54.33|67.39|70.48|96.34|76.15|49.46|68.15|68.09|90.78|65.22|52.39
MMVG-B-Lora(Hi-Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |73.84|74.57|91.30|70.46|56.35|71.73|70.36|96.02|76.39|49.28|68.55|67.48|90.53|64.69|51.64|
MMVG-B-Lora(Dual-lora) lora-rgb adapter for text (warm-up)| ACMMM'24 | CLIP-B/CLIP-B |70.72|74.20|90.00|72.34|55.94|67.93|69.31|95.45|76.39|47.50|67.27|66.37|91.03|63.80|49.17|
MMVG-B-Lora(Dual-lora) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |72.03|74.29|89.76|72.34|56.04|66.30|69.02|95.53|75.57|46.79|68.55|67.26|91.16|64.90|50.47|
MMVG-B-Lora(lora-ir-only) no-lora for text| ACMMM'24 | CLIP-B/CLIP-B |73.84|73.68|90.47|68.90|54.93|70.10|69.23|95.12|76.06|47.59|68.55|67.48|91.79|64.33|50.94|
MMVG-B-Lora(As-Hi-Dual-lora_8_32) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |73.35|73.87|90.71|71.40|54.53|69.02|69.83|95.04|75.41|49.11|68.63|67.15|90.65|63.76|51.17|
MMVG-B-Lora(As-Hi-Dual-lora_16_64) lora-rgb adapter for text| ACMMM'24 | CLIP-B/CLIP-B |73.02|73.97|89.76|71.09|55.94|68.47|69.83|95.77|76.48|48.44|68.63|67.50|90.02|64.78|51.60|
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint| ACMMM'24 | CLIP-B/CLIP-B |TODO-continue
MMVG-B-Lora(Hi-Dual-lora) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |73.02|73.59|88.21|72.65|56.14|69.56|70.19|95.04|77.38|49.42|68.47|67.17|89.77|64.73|51.25|
MMVG-B-Lora(As-Hi-Dual-lora)(16,48) merged_pretrain_checkpoint w/o multi_stage model load| ACMMM'24 | CLIP-B/CLIP-B |73.68|73.31|89.40|70.15|54.93|69.56|70.50|95.04|77.22|49.77|68.39|67.50|90.27|65.02|51.64|
Ours  | - | CLIP-B/CLIP-B ||||||

### Training Dataset: Single RGBTVG_Mixup, ImageSize: 224, Modality: RGB
* val and test datasets orders：Mixup, RefFLIR, RefM$^3$FD, RefMFAD
  
|Methods| Venue |Visual/Language Backbone| flir_val|flir_test|flir_testA|flir_testB|flir_testC|m3fd_val|m3fd_test|m3fd_testA| m3fd_testB | m3fd_testC |mfad_val  | mfad_test  | mfad_testA | mfad_testB | mfad_testC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
CLIP-VG | TMM'23| CLIP-B/CLIP-B| |||||
HiVG-B (Warmup) | ACMMM'24 | CLIP-B/CLIP-B |||||||||||||
HiVG-B | ACMMM'24 | CLIP-B/CLIP-B |68.91|68.58|88.92|57.03|46.37|63.04|65.03|94.56|68.66|40.96|64.57|64.26|91.16|60.74|46.39|
Ours  | - | CLIP-B/CLIP-B ||||||

### More Idea About CLIP Model （VLP Model）
* 调整模型适配性：修改位置编码为 “可学习的动态编码”（如 ViTAR 的模糊位置编码），或调整 patch 大小（如用 32×32 patch 处理 640×640，得到 20×20=400 个 patch，更接近 196 的数量级）；
* 多尺度微调：先在 224×224 上微调（稳定语义特征），再用 336×336、448×448 逐步过渡到 640×640，降低分布偏移的冲击；


 ## News

- :fire: **Update on 2025/01/30: The full code and models of HiVG have been released!**
- :fire: **Update on 2024/12/28: We conducted a survey of Visual Grounding over the past decade, entitled "Towards Visual Grounding: A Survey" ([Paper](https://arxiv.org/pdf/2412.20206), [Project](https://github.com/linhuixiao/Awesome-Visual-Grounding)), Comments are welcome !!!**
- :fire: **Update on 2024/10/10: Our new grounding work **OneRef** ([Paper](https://arxiv.org/abs/2410.08021), [Code](https://github.com/linhuixiao/OneRef)) has been accepted by the top conference NeurIPS 2024 !**
- :fire: **Update on 2024/07/16: Our grounding work HiVG ([Paper](https://openreview.net/pdf?id=NMMyGy1kKZ), [Code](https://github.com/linhuixiao/HiVG)) has been accepted by the top conference ACM MM 2024 !**
- **Update on 2024/04/20: Release the HiVG project repository.**
- **Update on 2023/9/25: Our preliminary work CLIP-VG ([Paper](https://ieeexplore.ieee.org/abstract/document/10269126), [Code](https://github.com/linhuixiao/CLIP-VG)) has been accepted by the top journal IEEE Transaction on Multimedia (2023)!** 

 ## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@inproceedings{xiao2024hivg,
      title={HiVG: Hierarchical Multimodal Fine-grained Modulation for Visual Grounding},
      author={Linhui Xiao and Xiaoshan Yang and Fang Peng and Yaowei Wang and Changsheng Xu},
      booktitle={ACM Multimedia 2024},
      year={2024},
      url={https://openreview.net/forum?id=NMMyGy1kKZ}
}
``` 

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgments](#acknowledgments)


## Highlight
- **A concise hierarchical multimodal modulation framework**, which utilizes the hierarchical structure to gradually adapt CLIP to grounding. HiVG achieves fine-grained interaction between multi-level visual representations and language semantics, and significantly alleviates the task gap between CLIP and grounding.
- **The first to propose the hierarchical multimodal low-rank adaptation paradigm.** Hi LoRA is a basic and concise hierarchical adaptation paradigm, which is task-agnostic.
- **Extensive experiments are conducted to verify the effectiveness of HiVG approaches.** Results show that our method achieves promising results, surpassing the SOTA methods under the same setting by a significant margin. Besides, our model offers significant computing efficiency advantages.


## TODO
- [x] Release all the checkpoints.
- [x] Release the full model code, training and inference code.




## Introduction

Visual grounding, which aims to ground a visual region via natural language, is a task that heavily relies on cross-modal 
alignment. Existing works utilized uni-modal pre-trained models to transfer visual/linguistic knowledge separately while 
ignoring the multimodal corresponding information. Motivated by recent advancements in contrastive language-image 
pre-training and low-rank adaptation (LoRA) methods, we aim to solve the grounding task based on multimodal pre-training.
However, there exists significant task gaps between pre-training and grounding. Therefore, to address these gaps, we 
propose **a concise and efficient hierarchical multimodal fine-grained modulation framework**, namely **HiVG**. Specifically,
HiVG consists of a multi-layer adaptive cross-modal bridge and a hierarchical multimodal low-rank adaptation (Hi LoRA) 
paradigm. The cross-modal bridge can address the inconsistency between visual features and those required for grounding,
and establish a connection between multi-level visual and text features. Hi LoRA prevents the accumulation of perceptual 
errors by adapting the cross-modal features from shallow to deep layers in a hierarchical manner. Experimental results 
on five datasets demonstrate the effectiveness of our approach and showcase the significant grounding capabilities as well 
as promising energy efficiency advantages.

For more details, please refer to [our paper](https://arxiv.org/abs/2404.13400).

## Usage
### Dependencies
- Python 3.9.10
- Pytorch 2.2.2
- transformers==4.30.0
- peft==0.3.0
- Check [requirements.txt](requirements.txt) for other dependencies. 
- It is recommended that the code be run under Anaconda env. If a library is missing while the code is running, 
  you can simply install it using `pip install <library_name>` or `conda install <library_name>`.

Our model is **easy to deploy** in a variety of environments and **has been successfully tested** on multiple pytorch versions.

❗❗❗️
**(Updated April 15, 2025) Please note that some researchers tested HiVG models in the latest peft library and found 
that the CLIP model weights did not match, which reduced the accuracy. To solve this problem, you only need to ensure 
the peft version is 0.3.0.**


### Image Data Preparation
1.You can download the images from the original source and place them in your disk folder, such as `$/path_to_image_data`:
- [MS COCO 2014](download_mscoco2014.sh) (for RefCOCO, RefCOCO+, RefCOCOg dataset, almost 13.0GB) 
- [ReferItGame](https://drive.google.com/drive/folders/1D4shieeoKly6FswpdjSpaOrxJQNKTyTv)
- [Flickr30K Entities](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in)

   We provide a script to download the mscoco2014 dataset, you just need to run the script in terminal with the following command:
   ```
   bash download_mscoco2014.sh
   ```
   Or you can also follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md).

Only the image data in these datasets is used, and these image data is easily find in similar repositories of visual grounding work, such as [TransVG](https://github.com/linhuixiao/TransVG) etc. 
Finally, the `$/path_to_image_data` folder will have the following structure:  

```angular2html
|-- image_data
   |-- Flickr30k
      |-- flickr30k-images
   |-- other
      |-- images
        |-- mscoco
            |-- images
                |-- train2014
   |-- referit
      |-- images
```
- ```$/path_to_image_data/image_data/Flickr30k/flickr30k-images/```: Image data for the Flickr30K dataset, please download from this [link](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in). Fill the form and download the images.
- ```$/path_to_image_data/image_data/other/images/```: Image data for RefCOCO/RefCOCO+/RefCOCOg, i.e., mscoco2014. 
- ```$/path_to_image_data/image_data/referit/images/```: Image data for ReferItGame.

## Text-Box Anotations 
The labels are consistent with previous works such as [TransVG](https://github.com/linhuixiao/TransVG). **However, 
this paper employs contrastive learning and shuffles the training examples; therefore, 
you will need to re-download the data from us. Additionally, we also provide the `mixup` dataset for mixup grounding training, 
which comprises by the five training sets (i.e., RefCOCO/+/g, ReferIt, Flickr30k). Note that the RefCOCOg-g (i.e., gref) 
training set is excluded in the `mixup` because it exists test set data leakage. The val and test split in `mixup` are 
copied from the RefCOCOg dataset.**


### text-box anotations download
<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    <th style="text-align:center" > Mixup pretraining </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> url, size </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="https://drive.google.com/file/d/1oaKlHeEECr-KFSDcWUG3X0UNUhqjGugr/view?usp=drive_link">ref_data_shuffled</a>,  267.0MB </th>  <!-- table head -->
    </tr>
</table>

Download the above annotations to a disk directory such as `$/path_to_split/ref_data_shuffled`; then will have the following similar directory structure:

```angular2html
|-- /ref_data_shuffled
    ├── flickr
    │   ├── flickr_test.pth
    │   ├── flickr_train.pth
    │   └── flickr_val.pth
    ├── gref_umd
    │   ├── gref_umd_test.pth
    │   ├── gref_umd_train.pth
    │   └── gref_umd_val.pth
    ├── referit
    │   ├── referit_test.pth
    │   ├── referit_train.pth
    │   └── referit_val.pth
    ├── unc
    │   ├── unc_testA.pth
    │   ├── unc_testB.pth
    │   ├── unc_train.pth
    │   └── unc_val.pth
    ├── unc+
    │   ├── unc+_testA.pth
    │   ├── unc+_testB.pth
    │   ├── unc+_train.pth
    │   └── unc+_val.pth
    └── mixup
        ├── mixup_test.pth
        ├── mixup_train.pth
        └── mixup_val.pth
```


## Pre-trained Checkpoints

The checkpoints include the Base model and Large mode under the single-dataset fine-tuning setting and dataset-mixed 
grounding pretraining setting. 

### Single-dataset fine-tuning checkpoints download

<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> base model </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="https://drive.google.com/file/d/1vM_568M7DwnYmjEiJgXRnrDL5UT65CGJ/view?usp=drive_link"> finetuning_base (for all), ~4.0 GB </a>  </th>  <!-- table head -->
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> Large model </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="https://drive.google.com/file/d/1Yw_AVaYnw4amPsemFwKFurXgaKvJ11CB/view?usp=drive_link">finetuning_large (for all), ~8.0 GB </a>  </th>  <!-- table head -->
    </tr>
</table>




### Mixup grounding pre-training checkpoints download

<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > Mixup </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> base model </th> <!-- table head -->
        <th style="text-align:center" colspan="1"> <a href="https://drive.google.com/file/d/1TzDLWjS-lXEr2M9uwaSBlU0MRmaRLSmN/view?usp=sharing">mixup_pretraining_base, ~1.0 GB </a>  </th>  <!-- table head -->
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Large model </th>
    <th style="text-align:center" > <a href="https://drive.google.com/file/d/1H_tv9QcDK712Ie9flLgSCZmmj0HEcjb8/view?usp=drive_link">mixup_pretraining_large, ~2.0 GB</a> </th>
    </tr>
</table>


After downloading all of these checkpoints, you can save them in the following directory, allowing you to train and test 
the five datasets at once and just using a single script.

```angular2html
|-- /finetuning_checkpoints (base or large model)
    ├── flickr
    │   └── best_checkpoint.pth
    ├── gref_umd
    │   └── best_checkpoint.pth
    ├── referit
    │   └── best_checkpoint.pth
    ├── unc
    │   └── best_checkpoint.pth
    └── unc+
        └── best_checkpoint.pth

|-- /mixup_grounding_pretraining (base or large model)
    └── mixup
        └── best_checkpoint.pth
```



### CLIP domain generalized checkpoints download

Due to the domain bias of CLIP on the MSCOCO dataset, we follow previous work, such as TransVG++, VG-LAW, etc., to conduct 
pre-training for the backbone network on the MSCOCO dataset while excluding RefCOCO/+/g related images. 
For this pre-training, the [Detectron2](https://github.com/facebookresearch/detectron2) framework is used for detection and segmentation training under the vanilla LoRA paradigm. 
If you want to training HiVG, please download the fine-tuned CLIP model using LoRA on MSCOCO dataset from the link below.


<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Model </th>
    <th style="text-align:center" > Debiased CLIP model using LoRA on the MSCOCO dataset </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> base model (ViT-B/224) </th> <!-- table head -->
        <th style="text-align:center" colspan="1"> <a href="https://drive.google.com/file/d/1pgso4gjHselrj4ExqJP3PYRbbX754aRq/view?usp=sharing">clip_b_ml_cascade_maskrcnn_model_224, 580 MB </a>  </th>  <!-- table head -->
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Large model (ViT-L/224) </th>
    <th style="text-align:center" > <a href="https://drive.google.com/file/d/18T4g6P-duKifx5Ksw6gHmL0ttKW39Wa6/view?usp=sharing">clip_l_ml_cascade_maskrcnn_model_224, 1.6 GB</a> </th>
    </tr>
</table>

Alternatively, you can also use the original CLIP Hugging Face model for training, for which we provide a download link.
In this case, the performance may be degraded.

<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Model </th>
    <th style="text-align:center" > original CLIP Hugging Face model </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> base model (ViT-B/224) </th> <!-- table head -->
        <th style="text-align:center" colspan="1"> <a href="https://drive.google.com/file/d/1SgWSK6vOKgPpEaULlHGZBnxotZ241phG/view?usp=drive_link">clip-vit-base-patch16, 375 MB </a>  </th>  <!-- table head -->
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Large model (ViT-L/224) </th>
    <th style="text-align:center" > <a href="https://huggingface.co/openai/clip-vit-large-patch14/tree/main">clip-vit-large-patch14, 1.6 GB</a> </th>
    </tr>
</table>


## Training and Evaluation


### Evaluation


1. Download the images and text annotations for the five datasets, as well as the trained HiVG model and CLIP initialization model. 
   You need to change the ```$/path_to_clip``` in [models/HiVG.py](models/HiVG.py) to your ```original CLIP Hugging Face model``` CLIP model directory.

2. The evaluation script are as follows:
    ```angular2html
    |-- /train_and_eval_script
        ├── eval_single_dataset_finetuning_base.sh
        ├── eval_single_dataset_finetuning_large.sh
        ├── eval_mixup_grounding_pretraining_base.sh
        └── eval_mixup_grounding_pretraining_large.sh
    ```

3. You just need to change ```$/path_to_split```, ``` $/path_to_image_data```, ``` $/path_to_output``` to your own file directory to execute the above command.
   We strongly recommend to use the following commands to training or testing with different datasets and splits, which will significant reduce the training workforce. Such as:
    ```
    bash train_and_eval_script/eval_single_dataset_finetuning_base.sh
    ```

4. For a specific dataset, the instruction is just like follows:
    ```
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --save_hilora_clip --hi_lora_stage 3 --data_root /path_to_image_data --split_root /path_to_split/ref_data_shuffled --eval_model /patch_to_output/finetuning_base/unc/best_checkpoint.pth      --eval_set testA  --output_dir /patch_to_output/finetuning_base/unc;
    ```
    Please refer to the files in [train_and_eval_script](train_and_eval_script) for evaluation commands on other splits or datasets under different settings.

5. If you need to save the CLIP model for the current stage, you need to use flags ```--save_hilora_clip```

### Training

1. Download the images and text annotations for the five datasets, as well as the trained HiVG model and CLIP initialization model. 
   You need to change the ```$/path_to_clip``` in [models/HiVG.py](models/HiVG.py) to your ```original CLIP Hugging Face model```  CLIP model directory.

2. The evaluation script are as follows:
    ```angular2html
    |-- /train_and_eval_script
        ├── train_single_dataset_finetuning_base.sh
        ├── train_single_dataset_finetuning_large.sh
        ├── train_mixup_grounding_pretraining_base.sh
        └── train_mixup_grounding_pretraining_large.sh
    ```

3. You just need to change ```$/path_to_split```, ``` $/path_to_image_data```, ``` $/path_to_output``` to your own file directory to execute the above command.
   We strongly recommend to use the following commands to training or testing with different datasets and splits, which will significant reduce the training workforce. Such as:
    ```
    bash train_and_eval_script/train_single_dataset_finetuning_base.sh
    ```

4. **Notably, for a specific dataset, if you want to enable HiLoRA, your training may involve 4 stages: the warmup stage, 
   HiLoRA stage 1, HiLoRA stage 2, and HiLoRA stage 3.**

   **In the warm-up phase, MACA is not turned on, only the fusion Transformer encoder is trained, and HiLoRA training is 
   not turned on for the CLIP model. Note that during the loading process of multiple rounds of HiLoRA training, 
   CLIP needs to be loaded separately. This will cause some parameters to mismatch, which is normal.**

   **Note that the essence of the HiLoRA mechanism is a process of decomposing parameter learning, and its effectiveness
   is influenced by the learning rate and the number of epochs. Therefore, HiLoRA requires different learning rates and numbers of epochs at various stages for specific model 
   configurations. If you do not need to enable HiLoRA, simply leave `args.hi_lora_stage=0` as the default.** 

5. **The Large version of the model is somewhat difficult to train and empirically requires one or two stages of warmup.** 
   In the first stage, `arg.warmup` needs to be enabled, and the visual adapt layer must be forced to be empty `[]` 
   to train the cross-modal fusion encoder, which is equivalent to freezing the CLIP model. 
   Only 5-10 epochs are needed for this phase. In the second stage, `arg.warmup` is turned off, and normal training 
   is performed; at this time, linguistic information can fine-tune the visual features through the cross-modal bridge.

   Please refer to the files in [train_and_eval_script](train_and_eval_script) for training commands on other splits or datasets under different settings.


## Results

### 1. RefCOCO, RefCOCO+, RefCOCOg, ReferIt, Flickr, datasets
<details open>
<summary><font size="4">
SOTA Result Table
</font></summary>
<img src="docs/sota.jpg" alt="COCO" width="100%">
</details>

**(1) When compared to the CLIP-based fine-tuning SOTA work**, i.e., Dynamic-MDETR, our approach consistently 
outperforms it by achieving an increase of 3.15%(testB), 3.11%(testA), 4.30%(test), 5.55%(test), 
0.22%(test) on all five datasets. 

**(2) When compared to the detector-based fine-tuning SOTA work**, i.e., 
TransVG++, our approach demonstrates superior performance (improved by 2.30%(testB), 4.36%(testA), 2.49%(test), 
1.22%(test), 0.62%(test)) across all five datasets. The improvement of our results on the RefCOCO+/g datasets is 
considerably more significant, indicating our model exhibits a stronger capacity for semantic comprehension in complex 
sentences. 

**(3) When compared with the dataset-mixed pre-training works**, the base model of our work outperforms 
Grounding-DINO by 1.24%(testB), 1.81%(testA), and 1.68%(testA) on the RefCOCO/+/g 
datasets, and it also outperforms OFA by 3.93%(testB), 2.06%(testA), and 4.31%(testA). 
After dataset-mixed pre-training, our performance has significantly improved, further demonstrating the effectiveness 
of our method.

### 2. Our model also has significant energy efficiency advantages.

<details open>
<summary><font size="4">
Illustration
</font></summary>
<div align=center>
<img src="docs/result_performance.jpg" alt="COCO" width="100%"></div>
</details>

**Comparison between HiVG (base) and SOTA models, as well as the ablation study of HiVG on the main modules.** (a) HiVG 
achieves significant energy efficiency advantages, **8.2x** faster than TransVG++ while
outperforming it on RefCOCO-val. (b) The computational complexity of HiVG is **only 13.0%** compared with 
TransVG++. (c) HiVG outperforms SOTA models in different expression lengths on RefCOCOg-test. (d) Hi LoRA method brings
significant performance gains to HiVG model.


## Methods 

<p align="center"> <img src='docs/motivation.jpg' align="center" width="60%"> </p>

**Visual attentions and grounding results of CLIP and the proposed HiVG.** The attentions are perceived by the 
[CLS] token over vision tokens.

<p align="center"> <img src='docs/hilora.jpg' align="center" width="60%"> </p>

**Hi LoRA and vanilla LoRA.** (a) The vanilla LoRA learns the global low-rank matrix utilizing the entire set of 
pre-trained weights in a single round. (b) The proposed Hi LoRA employs a hierarchical approach to adapt the pre-trained 
model in a progressive manner, thereby finely reducing the task gap between pre-training and transfer tasks.

## Visualization
<p align="center"> <img src='docs/visualization.jpg' align="center" width="70%"> </p>

 **Qualitative results of our HiVG framework on the RefCOCOg-val split.** The CLIP-VG model is compared. We present the
 prediction box with IoU (in cyan) and the ground truth box (in green) in a unified  image to visually display the 
 grounding accuracy. We show the [REG] token’s attention over vision tokens from the last 
 grounding block of each framework. The examples exhibit the relatively more challenging instances for grounding, thereby 
 showcasing HiVG's robust semantic comprehension capabilities.

## Contacts
Email: <xiaolinhui16@mails.ucas.ac.cn>.
Any kind discussions are welcomed!

## Acknowledgement

Our model is related to [CLIP](https://github.com/openai/CLIP), [CLIP-VG](https://github.com/linhuixiao/CLIP-VG). Thanks for their great work!

We also thank the great previous work including [TransVG++](https://github.com/linhuixiao/TransVG), 
[DETR](https://github.com/facebookresearch/detr), [QRNet](https://github.com/LukeForeverYoung/QRNet), etc. 

Thanks [OpenAI](https://github.com/openai) for their awesome models.




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=linhuixiao/HiVG&type=Date)](https://star-history.com/#linhuixiao/HiVG&Date)





