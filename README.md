# Supplementary materials for the paper Debias-SparseGPT: Bias-Aware Pruning for Large Language Models

🚧 Work in Progress 🚧

This repository will contain the code for our paper:
`Debias-SparseGPT: Bias-Aware Pruning for Large Language Models'.
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

### Evaluation

To evaluate perplexity, we rely on the implementation from the `evaluate` package, using `perplexity.compute` on the WikiText-2 dataset.  

For evaluating performance on the BBQ and UnQover benchmarks, we use the implementation from [FairSteer](https://github.com/LiYichen99/FairSteer).  

To evaluate MMLU, HellaSwag, and CrowS-Pairs scores, we use the [lm-eval-harness framework](https://github.com/EleutherAI/lm-evaluation-harness).  

To evaluate throughput, we use the Optimum benchmark.  

Carbon emissions are estimated using the equation (Impact Tracker, Henderson et al., 2020):  
CO2e = Energy (kWh) × CarbonIntensity (kgCO2e/kWh),  

where the carbon intensity reflects the electricity mix of the region in which training or inference is performed.  

For the Qwen model, we use the average carbon intensity for China, where the model was trained, as an example.  
The intensity is computed as the average over 2025:  
https://app.electricitymaps.com/map/live/fifteen_minutes.

## Hardware

All experiments are conducted on two NVIDIA A100 GPUs with 80 GB of memory each.
