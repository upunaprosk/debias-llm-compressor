# [EMNLP 2026 | Debias-SparseGPT: Bias-Aware Pruning for Large Language Models](https://openreview.net/forum?id=EyyWnQwQpV)

<a href="https://openreview.net/forum?id=EyyWnQwQpV" target="_blank">
  <img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" />
</a>
<a href="https://2026.emnlp.org/" target="_blank">
  <img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP-2026-red" />
</a>
<a href="https://github.com/vllm-project/llm-compressor" target="_blank">
  <img alt="LLM Compressor" src="https://img.shields.io/badge/⚙️-LLM--Compressor-blue" />
</a>

Official implementation and supplementary materials for **Debias-SparseGPT: Bias-Aware Pruning for Large Language Models**.

**Debias-SparseGPT** is a post-training pruning method that reduces **pruning-induced bias** in large language models while preserving model quality and the computational benefits of sparsification.

It supports both **unstructured sparsity** and hardware-friendly **2:4 semi-structured sparsity**, and is implemented directly on top of the [`llm-compressor`](https://github.com/vllm-project/llm-compressor) framework.

Authors: **Irina Proskurina, Guillaume Metzler, Antoine Gourru, and Julien Velcin**

---
🚧 Work in Progress 🚧

This repository will contain the code for our paper:
`Debias-SparseGPT: Bias-Aware Pruning for Large Language Models'.

---

## 🔥 News

* **August 2026:** 🎉 **Debias-SparseGPT was accepted to the EMNLP 2026 Main Conference!**
* **August 2026:** 🚀 We release the official implementation and reproducibility code.
* More models, recipes, and evaluation utilities coming soon.

--

Debias-SparseGPT is implemented as an additional method in the llm-compressor (https://github.com/vllm-project/llm-compressor) package.  
Compression is performed by specifying the recipe (`.*yaml`) file and calling the DebiasSparseGPTModifier method.  
To compress a model, fork the llm-compressor v0.8.1 version and copy the files provided in the debias-sparsegpt folder:

```
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
git checkout v0.8.1
cd ..
cp debias_sparsegpt_ss/datasets* llm-compressor/llm-compressor/datasets/
cp debias_sparsegpt_ss/modifiers* llm-compressor/llm-compressor/modifiers/
pip install -r llm-compressor/requirements.txt
```

To run the compression, use the provided script `debias_sparse_llama.py` for semi-structured pruning and `debias_sparse_llama_unstr.py` for unstructured pruning.  
The `--model` argument can be provided for any model on Hugging Face or a local folder with downloaded weights.

```
python debias_sparse_llama.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir output_llama8b_2of4 \
  --recipe 1_4_sparse_recipe.yaml
```

For unstructured sparsification (used in §5.2):

```
python debias_sparse_llama_unstr.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir output_llama8b_50 \
  --recipe 50_recipe.yaml
```

Supported recipes:
- 1_4_sparse_recipe.yaml
- 2_4_sparse_recipe.yaml
- 25_recipe.yaml
- 50_recipe.yaml

## Calibration data

We use the StereoSet development subset:
https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json

(Gonçalves & Strubell, EMNLP 2023)

For the 2:4 setting (§5.2), we also use UltraChat (Ding et al., 2023).

```
python debias_sparsegpt_ultrachat.py \
  meta-llama/Llama-3.1-8B-Instruct \
  2:4 \
  100% \
  256
```

## Evaluation

We evaluate Debias-SparseGPT across both **performance** and **fairness** benchmarks.

### Perplexity

Perplexity is evaluated on **WikiText-2** using the Hugging Face `evaluate` package and `perplexity.compute`.

### Fairness

We evaluate bias using:

* **BBQ**
* **UnQover**
* **CrowS-Pairs**

For BBQ and UnQover, we build on the implementation from [FairSteer](https://github.com/LiYichen99/FairSteer).

CrowS-Pairs evaluation is performed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Downstream Performance

We evaluate general model performance using:

* **MMLU**
* **HellaSwag**

These experiments are also conducted using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Throughput

Inference throughput is benchmarked using **Optimum Benchmark**.

---

## Carbon Emissions

Carbon emissions are estimated following **Impact Tracker** (Henderson et al., 2020):

$$
\mathrm{CO_2e}
=
\mathrm{Energy\ (kWh)}
\times
\mathrm{Carbon\ Intensity\ (kgCO_2e/kWh)}
$$

Carbon intensity corresponds to the electricity mix of the geographical region in which training or inference is performed.

For the Qwen experiments, we use China as an illustrative regional estimate and compute the carbon intensity from the **2025 annual average** reported by Electricity Maps:

```text
https://app.electricitymaps.com/map/live/fifteen_minutes
```

---

## Hardware

All experiments reported in the paper were conducted using:

**2 × NVIDIA A100 GPUs with 80 GB of memory each.**

---

## Citation

If you find Debias-SparseGPT useful in your research, please consider citing our paper (to appear in the Proceedings of EMNLP 2026)
---

## ⭐ Support

If you find this repository useful, consider giving it a **⭐ star** — it helps others discover the project.

Questions, bug reports, and suggestions are welcome through GitHub Issues.

