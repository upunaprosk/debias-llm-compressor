# Debias-SparseGPT

<p align="center">
  <strong>Bias-aware post-training pruning for large language models</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.02496">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.02496-b31b1b.svg">
  </a>
  <a href="https://openreview.net/forum?id=EyyWnQwQpV">
    <img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-Paper-8c1b13">
  </a>
  <a href="https://2026.emnlp.org/">
    <img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP-2026-red">
  </a>
  <a href="https://github.com/vllm-project/llm-compressor">
    <img alt="LLM Compressor" src="https://img.shields.io/badge/LLM--Compressor-0.8.1-blue">
  </a>
  <a href="https://github.com/upunaprosk/debias-llm-compressor/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/upunaprosk/debias-llm-compressor/actions/workflows/tests.yml/badge.svg?branch=development">
  </a>
</p>

Official implementation and reproducibility code for:

> **Debias-SparseGPT: Bias-Aware Pruning for Large Language Models**
> Irina Proskurina, Guillaume Metzler, Antoine Gourru, and Julien Velcin
> **EMNLP 2026 Main Conference**

Debias-SparseGPT is a post-training pruning method designed to reduce **pruning-induced social bias** while preserving model quality and the computational benefits of sparsification.

Debias-SparseGPT implementation follows the [`llm-compressor`](https://github.com/vllm-project/llm-compressor) compression framework.

---

## Installation

Clone the repository:

```
git clone https://github.com/upunaprosk/debias-llm-compressor.git
cd debias-llm-compressor
```

Create and activate a virtual environment:

```
python3.11 -m venv .venv
source .venv/bin/activate
```

Debias-SparseGPT uses a patched version of `llm-compressor 0.8.1`. First clone the corresponding upstream version:

```
git clone \
  --branch 0.8.1 \
  https://github.com/vllm-project/llm-compressor.git \
  third_party/llm-compressor
```

For the **StereoSet-only** setup, apply:

```
git -C third_party/llm-compressor apply \
  "$PWD/patches/llm_compressor_0.8.1_stereoset.patch"
```

For the **mixed StereoSet + UltraChat** setup, apply:

```
git -C third_party/llm-compressor apply \
  "$PWD/patches/llm_compressor_0.8.1_mixed_calibration.patch"
```

Install the patched backend:

```
BUILD_TYPE=release \
python -m pip install -e third_party/llm-compressor
```

Then install Debias-SparseGPT without replacing the dependency versions from `llm-compressor`:

```
python -m pip install --no-deps -e .
```

---

## Command-line interface

```
debias-sparsegpt --help
```

In the paper, we experiment with 1) stereoset-only calibration, and 2) mixed ultrachat-stereoset calibration:

```
debias-sparsegpt stereoset
debias-sparsegpt mixed
```

### Shared Arguments

| Argument     | Default                            | Description                                                  |
| ------------ | ---------------------------------- | ------------------------------------------------------------ |
| `--model`    | `meta-llama/Llama-3.1-8B-Instruct` | HF model or path to a local checkpoint |
| `--recipe`   | required                           | Path to the `llm-compressor` sparsity recipe                |
| `--sparsity` | `2:4`                              | Sparsity structure. Supported values are `1:4` and `2:4`    |
| `--alpha`    | `0.0`                              | Weight of the bias-aware Debias-SparseGPT term. In the paper, we use 1 and 0 values             |
| `--seed`     | `1`                                | Random seed.                                                |
| `--workers`  | `4`                                | Number of preprocessing workers                           |

The `ALPHA` environment variable is also supported:

```
ALPHA=0.1 debias-sparsegpt stereoset ...
```

---

## StereoSet calibration

The `stereoset` reproduces the StereoSet-only calibration setup:

```
debias-sparsegpt stereoset \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --sparsity "2:4" \
  --alpha 0.1 \
  --output-dir output_llama8b_2of4
```

Additional arguments:

| Argument           | Default               | Description                                                                                  |
| ------------------ | --------------------- | -------------------------------------------------------------------------------------------- |
| `--stereoset`      | none                  | Optional path to a local StereoSet `dev.json`. If omitted, the original data source is used. |
| `--output-dir`     | `output_llama8b_2of4` | Output directory                                            |
| `--max-seq-length` | `100`                 | Maximum sequence length used for StereoSet calibration                                      |

Example:

```
debias-sparsegpt stereoset \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --sparsity "2:4" \
  --alpha 0.1 \
  --seed 1 \
  --workers 4 \
  --stereoset data/stereoset/dev.json \
  --max-seq-length 100 \
  --output-dir results/stereoset
```

---

## Mixed StereoSet + UltraChat calibration

The `mixed` command reproduces the calibration setting in which StereoSet is combined with general-language calibration data from UltraChat.

```
debias-sparsegpt mixed \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --sparsity "2:4" \
  --alpha 0.1 \
  --stereoset-samples 1000 \
  --ultrachat-samples 256 \
  --output-dir output_models
```

Additional arguments:

| Argument                     | Default         | Description                                     |
| ---------------------------- | --------------- | ----------------------------------------------- |
| `--stereoset`                | none            | Optional path to a local StereoSet `dev.json`.  |
| `--stereoset-samples`        | all available   | Number of StereoSet calibration examples       |
| `--ultrachat-samples`        | `256`           | Number of UltraChat calibration examples       |
| `--stereoset-max-seq-length` | `64`            | Maximum sequence length for StereoSet examples |
| `--ultrachat-max-seq-length` | `1024`          | Maximum sequence length for UltraChat examples |
| `--output-dir`               | `output_models` | Base directory for saved checkpoints           |

Example:

```
debias-sparsegpt mixed \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --sparsity "2:4" \
  --alpha 0.1 \
  --seed 1 \
  --workers 4 \
  --stereoset data/stereoset/dev.json \
  --stereoset-samples 1000 \
  --ultrachat-samples 256 \
  --stereoset-max-seq-length 64 \
  --ultrachat-max-seq-length 1024 \
  --output-dir output_models
```

The mixed pipeline can also be run with only one of the two calibration datasets:

```
# StereoSet only
debias-sparsegpt mixed \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --alpha 0.1 \
  --stereoset-samples 1000 \
  --ultrachat-samples 0
```

```
# UltraChat only
debias-sparsegpt mixed \
  --recipe recipes/2_4_sparse_recipe.yaml \
  --alpha 0.1 \
  --stereoset-samples 0 \
  --ultrachat-samples 256
```

---

## Sparsity recipes

| Recipe                   | Sparsity                     |
| ------------------------ | ---------------------------- |
| `1_4_sparse_recipe.yaml` | 1:4 semi-structured sparsity |
| `2_4_sparse_recipe.yaml` | 2:4 semi-structured sparsity |
| `25_recipe.yaml`         | 25% unstructured sparsity    |
| `50_recipe.yaml`         | 50% unstructured sparsity    |

---

## Calibration data

We use the StereoSet **intrasentence development set** introduced by [Nadeem et al. (2021)](https://aclanthology.org/2021.acl-long.416/).  
UltraChat [Ding et al. (2023)](https://arxiv.org/abs/2305.14233) is used as general-language calibration data in the mixed setup.

---

## Testing

```
python -m pip install pytest ruff
```

Run the tests:

```
python -m pytest -v
```

Linting:

```
python -m ruff check src tests
```

---

## Evaluation

We evaluate Debias-SparseGPT-compressed models' language-modelling performance, performance on social bias and toxicity benchmarks, downstream task performance, and inference efficiency.

Perplexity is evaluated on **WikiText-2**.
Next, we report results on:
* BBQ
* UnQover
* CrowS-Pairs

The BBQ and UnQover evaluations build on [FairSteer](https://github.com/LiYichen99/FairSteer). CrowS-Pairs is evaluated with the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
General model performance is evaluated on:
* MMLU
* HellaSwag

These experiments also use the LM Evaluation Harness.

### Throughput

Inference throughput is measured with Optimum Benchmark.


## Carbon Emissions

Carbon emissions are estimated following **Impact Tracker** (Henderson et al., 2020):

$$
\mathrm{CO_2e}=\mathrm{Energy\ (kWh)}\times\mathrm{Carbon\ Intensity\ (kgCO_2e/kWh)}
$$

Carbon intensity corresponds to the electricity mix of the geographical region in which training or inference is performed.

For the Qwen experiments, we use China as an illustrative regional estimate and compute the carbon intensity from the **2025 annual average** reported by Electricity Maps:

```
https://app.electricitymaps.com/map/live/fifteen_minutes
```

---

## Hardware

All experiments reported in the paper were conducted using:

**2 × NVIDIA A100 GPUs with 80 GB of memory each.**
---

## Citation

If you use Debias-SparseGPT in your research, please cite:

```bibtex
@misc{proskurina2026debiassparsegpt,
  title        = {Debias-SparseGPT: Bias-Aware Pruning for Large Language Models},
  author       = {Irina Proskurina and Guillaume Metzler and Antoine Gourru and Julien Velcin},
  year         = {2026},
  eprint       = {2609.02496},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2609.02496}
}
```

---

## ⭐ Support

If you find this repository useful, consider giving it a **⭐ star** — it helps others discover the project.

Questions, bug reports, and suggestions are welcome through GitHub Issues.
