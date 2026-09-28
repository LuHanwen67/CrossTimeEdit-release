<h1 align="center"><em>CrossTimeEdit</em><br><sub>A Decade-Spanning Cross-View Dataset and Reward-Guided Editing for Historical Street-View Generation</sub></h1>

<p align="center">
Hanwen Lu<sup>1*</sup> &nbsp; Jun He<sup>1*</sup> &nbsp; Mingjia Yang<sup>1</sup> &nbsp; Hao Wei<sup>1</sup> &nbsp; Jinhao Huang<sup>1</sup> &nbsp; Yi Lin<sup>1</sup> &nbsp; Xiang Zhang<sup>1&dagger;</sup><br>
<sub><sup>1</sup> Sun Yat-sen University</sub><br>
<sub>* Equal contribution &nbsp; &dagger; Corresponding author</sub>
</p>

<p align="center">
<a href="https://luhanwen67.github.io/CrossTimeEdit-release/"><img src="https://img.shields.io/badge/Project-Page-246b96?style=flat-square" alt="Project page"></a>
<a href="https://huggingface.co/LuHanwen/CrossTimeEdit-LoRA"><img src="https://img.shields.io/badge/Hugging_Face-Model-307c73?style=flat-square" alt="Model weights"></a>
<a href="https://huggingface.co/datasets/LuHanwen/VIGOR-his-metadata"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-96764d?style=flat-square" alt="Dataset metadata"></a>
</p>

<p align="center">
<a href="#abstract">Abstract</a> &middot;
<a href="#resources">Resources</a> &middot;
<a href="#quick-start">Quick start</a> &middot;
<a href="#training">Training</a> &middot;
<a href="#citation">Citation</a>
</p>

<a id="abstract"></a>

## 📝 Abstract

Historical street-view imagery records urban evolution, but uneven coverage
leaves substantial gaps in historical records. Generating plausible past
appearances requires restoring changed structures while preserving persistent
scene content. We construct VIGOR-his, a decade-spanning cross-view dataset
containing 43,653 location-level quadruplets across 11 cities on three
continents. Its automated pipeline performs spatial pairing, consistency
screening, change classification, and the generation and validation of
satellite-based change descriptions and local editing instructions. Based on
VIGOR-his, we propose CrossTimeEdit, a model that reformulates historical
street-view generation as editing, using recent street views to constrain
viewpoint and unchanged appearance and temporal satellite differences as
change evidence. Starting from FLUX.2 [Klein] 4B, we train CrossTimeEdit through
supervised fine-tuning (SFT) followed by online reinforcement learning (RL). We
design three street-view editing criteria, Instruction Alignment (IA),
Background Preservation (BP), and Quality and Physical Plausibility (QP), as
both RL reward dimensions and evaluation metrics. We optimize this multi-reward
objective using Within Group Relative Policy Optimization for flow-matching
models (Flow-GRPO) with Group reward-Decoupled Normalization Policy Optimization
(GDPO), which normalizes each reward dimension before aggregation. CrossTimeEdit
improves overall performance across the three editing criteria by 17.12% over
the pretrained baseline, ranks first among evaluated open-source image editing
models, and outperforms cross-view generation models in scene consistency,
visual realism, and perceptual quality.

![CrossTimeEdit teaser](assets/teaser.png)

*VIGOR-his provides paired earlier and recent street and satellite views, cross-view consistency screening, geographic coverage across 11 cities on three continents, and structured change annotations for historical street-view generation.*

## 🔎 At a Glance

| VIGOR-his | CrossTimeEdit |
| :--- | :--- |
| **43,653** location-level temporal cross-view quadruplets | **FLUX.2 [Klein] 4B** with LoRA adaptation |
| **11 cities** across **3 continents** | Supervised fine-tuning followed by online RL |
| Approximately **a decade** between observations | **IA, BP, and QP** rewards with GDPO-normalized Flow-GRPO |
| Change categories, satellite descriptions, and local editing instructions | **17.12%** improvement over the pretrained baseline across the three editing criteria |

<a id="resources"></a>

## 📦 Resources

| Resource | Release |
| :--- | :--- |
| Project page | [Figures, tables, and method overview](https://luhanwen67.github.io/CrossTimeEdit-release/) |
| Model weights | [CrossTimeEdit LoRA](https://huggingface.co/LuHanwen/CrossTimeEdit-LoRA), selected epoch-200 checkpoint |
| Dataset metadata | [VIGOR-his](https://huggingface.co/datasets/LuHanwen/VIGOR-his-metadata), including panorama identifiers and experiment splits |

> Google Maps and Street View imagery is **not redistributed**. Image acquisition remains subject to the provider's terms and availability. The released LoRA requires the base model, which must be obtained separately under its applicable terms.

<a id="quick-start"></a>

## 🚀 Quick Start

### ⚙️ 1. Set up the environment

Use Linux, Python 3.10 or 3.11, and a CUDA-compatible PyTorch installation.

```bash
git clone https://github.com/LuHanwen67/CrossTimeEdit-release.git
cd CrossTimeEdit-release
python3 -m venv .venv
source .venv/bin/activate
```

Install the PyTorch build appropriate for your CUDA version **inside this environment**, then run:

```bash
bash scripts/setup_sft.sh
pip install -e third_party/flow_factory
pip install -r environment/requirements-evaluation.txt
pip install huggingface_hub
```

### 📥 2. Download the release

```bash
hf download LuHanwen/CrossTimeEdit-LoRA crosstimeedit.safetensors --local-dir weights
hf download LuHanwen/VIGOR-his-metadata --repo-type dataset --local-dir data
```

Place the base-model files and legally obtained images in the following structure. Image filenames must match the relative paths in the manifests.

```text
models/FLUX.2-klein-base-4B/
data/
  metadata.jsonl
  manifests/
    train.json
    val.json
    test.json
  images/
weights/
  crosstimeedit.safetensors
```

### 🖼️ 3. Generate historical street views

Single-GPU inference:

```bash
python inference/generate.py \
  --gpu 0 --rank 0 --world-size 1 \
  --manifest data/manifests/test.json \
  --data-root data \
  --base-model models/FLUX.2-klein-base-4B \
  --lora weights/crosstimeedit.safetensors \
  --output-dir outputs/test
```

For multiple GPUs, run one process per GPU with the same `--world-size` and distinct `--gpu` and `--rank` values.

### 📊 4. Evaluate outputs

Set `GEMINI_API_KEY` in your environment, then run:

```bash
python evaluation/editing_vlm.py \
  --name CrossTimeEdit \
  --manifest data/manifests/test.json \
  --data-root data \
  --image-dir outputs/test \
  --output outputs/editing_scores.jsonl
```

The editing protocol evaluates **Instruction Alignment (IA)**, **Background Preservation (BP)**, and **Quality and Physical Plausibility (QP)**.

<a id="training"></a>

## 🧠 Training

<details>
<summary><strong>🎯 Supervised fine-tuning</strong></summary>

Prepare the SFT JSONL manifest for the patched DiffSynth-Studio training loader. The released split JSON files define split membership; they are not the SFT loader's JSONL input.

```bash
export MODEL_BASE="$(pwd)/models/FLUX.2-klein-base-4B"
export DATA_ROOT="$(pwd)/data"
export SFT_MANIFEST="$(pwd)/data/manifests/sft_train.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sft_phase1.sh
bash scripts/train_sft_phase2.sh
```

The launch scripts use four processes, LoRA rank 32, and 512 x 1024 panoramas.

</details>

<details>
<summary><strong>🔄 Online reinforcement learning</strong></summary>

Prepare the SFT initialization and RL data at the paths in [the main configuration](configs/rl/main.yaml), or update the configuration to match your local layout.

```text
weights/sft/v3_e9_lora/
data/rl/streetview_change_gt/
```

Set `GEMINI_API_KEY` in your environment, then launch:

```bash
export GEMINI_BASE_URL=https://generativelanguage.googleapis.com
bash scripts/train_rl.sh
```

The configuration uses Flow-GRPO with GDPO normalization and IA/BP/QP reward weights of **0.45 / 0.30 / 0.25**.

</details>

## 🗂️ Repository Guide

| Path | Purpose |
| :--- | :--- |
| [`configs/rl/main.yaml`](configs/rl/main.yaml) | Main Flow-GRPO/GDPO configuration |
| [`prompts/`](prompts/) | Dataset-construction and evaluation prompts |
| [`scripts/`](scripts/) | Environment setup and SFT/RL launch scripts |
| [`patches/`](patches/) | DiffSynth-Studio SFT patch |
| [`inference/generate.py`](inference/generate.py) | Historical panorama generation |
| [`evaluation/editing_vlm.py`](evaluation/editing_vlm.py) | IA/BP/QP evaluation |
| [`third_party/flow_factory/`](third_party/flow_factory/) | Main-experiment RL implementation |
| [`site/`](site/) | Project-page source |

<a id="citation"></a>

## 📖 Citation

```bibtex
@misc{lu2026crosstimeedit,
  title  = {CrossTimeEdit: A Decade-Spanning Cross-View Dataset and Reward-Guided Editing for Historical Street-View Generation},
  author = {Lu, Hanwen and He, Jun and Yang, Mingjia and Wei, Hao and Huang, Jinhao and Lin, Yi and Zhang, Xiang},
  year   = {2026},
  note   = {arXiv preprint}
}
```
