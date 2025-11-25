# NOTE: Fine tuning can require more steps than is shown in the example
# See the Axolotl integration blog post for best fine tuning practices
# https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open

import os

from pathlib import Path

import json
import numpy as np
import torch
import random
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot, train

from datasets import Dataset, load_dataset, concatenate_datasets

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Set random seed
seed_everything(1)

# load the model in as bfloat16 to save on memory and compute
# model_stub = "neuralmagic/Llama-2-7b-ultrachat200k"
# model_stub = os.path.join(os.environ['HOME'], "meta-llama/Llama-2-7b-chat-hf")
# model_stub = "meta-llama/Llama-3.2-1B-Instruct"
model_stub = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_stub, dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# uses LLM Compressor's built-in preprocessing for ultra chat
# dataset = "ultrachat-200k"
# dataset_name = dataset

# Load StereoSet data
# url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
import urllib.request

with urllib.request.urlopen(url) as response:
    data = json.load(response)

examples = []
for entry in data['data']['intrasentence']:
    # check if both stereotype and anti-stereotype exist
    check_set = set()
    X0 = None
    for sentence_entry in entry['sentences']:
        sentence = sentence_entry['sentence']
        gold_label = sentence_entry['gold_label']
        if gold_label in ['stereotype', 'anti-stereotype']:
            check_set.add(gold_label)
            if gold_label == 'anti-stereotype':
                X0 = sentence
    if len(check_set) < 2:
        continue
    # add pair of examples
    for sentence_entry in entry['sentences']:
        sentence = sentence_entry['sentence']
        gold_label = sentence_entry['gold_label']
        if gold_label in check_set:
            examples.append(sentence)
            #! dbg
            # examples.append(X0)
print(examples[:2])

# Select the recipe for 2 of 4 sparsity and 4-bit activation quantization
recipe = "2of4_recipe.yaml"

# save location of quantized model
output_dir = "output_qwen7b_2of4"
output_path = Path(output_dir)

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}
max_seq_length = 512
# num_calibration_samples = 512
dataset = Dataset.from_dict({"text": examples})
dataset_name = "StereoSet"
num_calibration_samples = len(examples)

preprocessing_num_workers = 4

oneshot_kwargs = dict(
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    splits=splits,
)
if dataset_name == "StereoSet":
    # StereoSet "intrasentence" sentences are short: they have max. 33 words
    oneshot_kwargs["max_seq_length"] = 100

# Oneshot sparsification

print(f"{num_calibration_samples=}")
print(f"Debias alpha = {os.environ.get('ALPHA', '')}")
sparse_model = oneshot(
    model=model,
    **oneshot_kwargs,
    # output_dir=output_dir,
    stage="sparsity_stage",
)

sparse_model.save_pretrained(
    f"{output_dir}/sparsity_stage", skip_sparsity_compression_stats=False,
    save_compressed=True,
    disable_sparse_compression=False, # it's the default, just in case
)
tokenizer.save_pretrained(f"{output_dir}/sparsity_stage")
