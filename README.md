<p align="center">
<!--   <h1 align="center"><img src="assets/logo.png" width="256"></h1> -->
  <h1 align="center">ScaleCap: Inference-Time Scalable Image Captioning
via Dual-Modality Debiasing</h1>
    <p align="center">
    <a href="https://github.com/Cooperx521"><strong>Long Xing*</strong></a>
    ·
    <a href="https://github.com/shikiw"><strong>Qidong Huang*</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a>
    ·
    <a href="https://panzhang0212.github.io/"><strong>Pan Zhang</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ"><strong>Yuhang Cao</strong></a>
    ·
    <a href="https://li-jinsong.github.io/"><strong>Jinsong Li</strong></a>
    ·
    <a href="https://mark12ding.github.io/"><strong>Shuangrui Ding</strong></a>
    ·
    <strong>Weiming Zhang</strong>
    ·
    <strong>Nenghai Yu</strong>
    ·
    <strong>Jiaqi Wang</strong>
    .
    <strong>Feng Wu</strong>
    .
    <strong>Dahua Lin</strong>
  </p>
  <!-- 📖<a href="https://arxiv.org/abs/2503.01785">Paper</a> |
  🤗<a href="https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df">Datasets</a> | 🤗<a href="https://huggingface.co/papers/2503.01785">Daily Paper</a></h3> -->
<div align="center"></div>
<p align="center">
  <p>
🌈We introduce <strong>ScaleCap</strong>, an <strong>inference-time scalable image captioning</strong> strategy that generates comprehensive and detailed image captions. With ScaleCap, we construct a dataset containing <strong>450k image-caption pairs</strong> for use by the open-source community. Our key observations highlight two <strong>inherent biases</strong> in LVLMs: <strong>multimodal bias</strong> resulting in <strong>imbalanced descriptive granularity</strong>; <strong>linguistic bias</strong> leading to hallucinated descriptions of non-existent objects. To address these issues, we propose two novel components: heuristic question answering and contrastive sentence rating. Extensive modality alignment experiments demonstrate the effectiveness of ScaleCap.

  </p>

<a href="">
  <img src="assets/teaser.png" alt="Logo" >
</a>

<a href="">
  <img src="assets/data.png" alt="Logo" >
</a>

## 📢 News
- 🚀 [06/25/2025] We release **ScaleCap** repository, training code and dataset.

## 💡 Highlights
- 🔥 **A plug-and-play pipeline improving caption quality**: ScaleCap can be used simply by calling either open-source or closed-source model APIs, making it extremely convenient to use.
- 🔥 **450k Image-Caption Dataset**: With ScaleCap, we construct a dataset containing 450k image-caption pairs for use by the open-source community.
- 🔥 **Extensive Experiments**: We conduct **extensive experiments** on various tasks to demonstrate effectiveness of ScaleCap.
- 🔥 **Open Source**: We fully **open-source** the training code, training data, and evaluation scripts on Github to facilitate further research.

<a href="">
  <img src="assets/pipeline.png" alt="Logo" >
</a>

## 🛠️ Setup
```
git clone https://github.com/Cooperx521/ScaleCap.git
conda create -n ScaleCap python=3.10
conda activate ScaleCap
bash setup.sh
```

## ⭐️ Quick Start
