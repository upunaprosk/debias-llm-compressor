import os
import sys
import json
import yaml
import torch
import random
import urllib
from loguru import logger
from copy import deepcopy
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset, load_dataset, concatenate_datasets

from torch.utils.data import DataLoader

from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.datasets.utils import get_calibration_dataloader

os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppress warning "Disabling parallelism to avoid deadlocks..."

# --- CombinedDataLoader helper (simple sequential wrapper) ---
class CombinedDataLoader:
    """Iterate multiple dataloaders sequentially inside a single calibration epoch.

    Note: This wrapper is intended for use with the `basic` calibration pipeline,
    which only needs an iterator and __len__ (streaming). Do NOT use this
    wrapper with the `sequential` pipeline (that one needs indexing / caching).
    """
    def __init__(self, dataloaders):
        self.dataloaders = list(dataloaders)
        # compute total length if possible
        try:
            self._len = sum(len(dl) for dl in self.dataloaders)
        except Exception:
            self._len = None

    def __iter__(self):
        for dl in self.dataloaders:
            for batch in dl:
                yield batch

    def __len__(self):
        if self._len is None:
            raise TypeError("length not available for one or more sub-dataloaders")
        return self._len

# Load StereoSet data
url = os.path.join(os.environ["HOME"], ".cache/stereoset/dev.json")
if not os.path.exists(url):
    url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
if url.startswith("http"):
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
else:
    with open(url, "rt") as file:
        data = json.load(file)
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
            # examples.append(X0) # debug sanity check: (X0, X0) => H_x01 == 0
print(examples[:2])

dataset = Dataset.from_dict({"text": examples})

dataset_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=None)

# model_stub = "meta-llama/Llama-3.2-1B-Instruct"
model_stub = "meta-llama/Llama-3.1-8B-Instruct"
model_name = model_stub.split("/")[-1]
if len(sys.argv) > 1:
    model_stub = sys.argv[1]
sparsity = "2:4" # OR "1:4"
if len(sys.argv) > 2:
    sparsity = sys.argv[2]
print(f"Loading {model_stub}...")
model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.bfloat16)
model.generation_config.do_sample = True # fix for vicuna-7b-v1.5
tokenizer = AutoTokenizer.from_pretrained(model_stub)
print(f"Loading {model_stub}... Done.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

# --- Prepare recipe ---
recipe = None
if sparsity == "2:4":
    recipe = "2of4_recipe.yaml"
    RECIPE_YAML = """
sparsity_stage:
  sparsity_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      mask_structure: "2:4"
      targets: ["Linear"]
      ignore: ["re:.*lm_head"]
"""
elif sparsity == "1:4":
    recipe = "1of4_recipe.yaml"
    RECIPE_YAML = """
sparsity_stage:
  sparsity_modifiers:
    SparseGPTModifier:
      sparsity: 0.25
      mask_structure: "1:4"
      targets: ["Linear"]
      ignore: ["re:.*lm_head"]
"""
else:
    print("Wrong sparsity structure: {sparsity=}")
    sys.exit(1)

print(f"Configuring {sparsity=}...")
if not os.path.exists(recipe):
    with open(recipe, "w") as f:
        f.write(RECIPE_YAML)
        f.flush()

oneshot_kwargs = dict(
    preprocessing_num_workers=4,
    num_calibration_samples=5, # will be redefined for each of the calibration datasets
    max_seq_length=100,        # will be redefined for each of the calibration datasets
)

# --- Instantiate Oneshot to configure args/session ---
# We pass minimal kwargs that Oneshot.parse_args expects: model, dataset (we provide ds_short
# just to populate dataset_args). Oneshot will set model_args, dataset_args, recipe_args on the
# instance. We'll override dataset_args.pipeline to "basic" to force BasicPipeline.
oneshot_inst = Oneshot(
    model=model,
    dataset=dataset_loader,  # only used to construct dataset_args; not used for calibration here
    recipe=recipe,
    **oneshot_kwargs,
)

# GitHub Copilot added this option:
# "Force basic pipeline to avoid sequential pipeline requirements"
# oneshot_inst.dataset_args.pipeline = "basic"
# but this causes VRAM OOM, so don't do this

# create DatasetArguments for StereoSet
args_ds1 = deepcopy(oneshot_inst.dataset_args)
args_ds1.splits = None # "all" by default
args_ds1.dataset = dataset
args_ds1.batch_size = 2
args_ds1.num_calibration_samples = len(examples) # full StereoSet
if len(sys.argv) > 3:
    if sys.argv[3] == "100%":
        args_ds1.num_calibration_samples = len(examples)
    else:
        args_ds1.num_calibration_samples = int(sys.argv[3])
args_ds1.num_calibration_samples = min(len(examples), args_ds1.num_calibration_samples)
args_ds1.max_seq_length = 64 # StereoSet "intrasentence" sentences are short: they have max. 33 words

# get tokenized datasets (returns dict with 'calibration')
loader1 = get_calibration_dataloader(args_ds1, processor=oneshot_inst.processor)

# create DatasetArguments for Ultrachat
args_ds2 = deepcopy(oneshot_inst.dataset_args)
args_ds2.splits = {"calibration": "train_gen[:1%]"} # 1024 < 1% of 200k
args_ds2.dataset = 'ultrachat-200k'
args_ds2.batch_size = 1
args_ds2.num_calibration_samples = 256
if len(sys.argv) > 4:
    args_ds2.num_calibration_samples = int(sys.argv[4])
args_ds2.num_calibration_samples = min(2000, args_ds2.num_calibration_samples)
args_ds2.max_seq_length = 1024

loader2 = get_calibration_dataloader(args_ds2, processor=oneshot_inst.processor)

print(f"Dataset(s):")
if args_ds1.num_calibration_samples > 0:
    print(f"StereoSet length={args_ds1.num_calibration_samples} batch_size={args_ds1.batch_size} max_seq_length={args_ds1.max_seq_length}")
if args_ds2.num_calibration_samples > 0:
    print(f"Ultrachat length={args_ds2.num_calibration_samples} batch_size={args_ds2.batch_size} max_seq_length={args_ds2.max_seq_length}")

if args_ds1.num_calibration_samples > 0 and args_ds2.num_calibration_samples > 0:
    combined_dataloader = CombinedDataLoader([loader1, loader2])
elif args_ds1.num_calibration_samples > 0:
    combined_dataloader = loader1
elif args_ds2.num_calibration_samples > 0:
    combined_dataloader = loader2
else:
    print("Zero num_calibration_samples, exitting...")
    sys.exit(1)

# Check of the combined data loader
# for i, s in enumerate(combined_dataloader):
#    print(f"{i=} {s['input_ids'].shape=}")

if "ALPHA" not in os.environ:
    os.environ["ALPHA"] = "0"
print(f"Debias alpha = {os.environ.get('ALPHA', '')}")

# --- Run calibration (this will initialize modifiers, run the pipeline, and finalize session) ---
# apply_recipe_modifiers accepts a prepared dataloader; pass our combined dataloader.
oneshot_inst.apply_recipe_modifiers(calibration_dataloader=combined_dataloader)

model.generation_config.do_sample = True # fix error at saving vicuna-7b-v1.5

output_dir = "output_models"
alpha = os.environ.get("ALPHA", "")
sp_name = "sparse" + sparsity.replace(":", "")
if args_ds1.num_calibration_samples > 0 and args_ds2.num_calibration_samples > 0:
    model_output_dir = f"{output_dir}/{model_name}-{sp_name}-stereo{args_ds1.num_calibration_samples}-ultrachat{args_ds2.num_calibration_samples}-alpha{alpha}"
elif args_ds1.num_calibration_samples > 0:
    model_output_dir = f"{output_dir}/{model_name}-{sp_name}-stereo{args_ds1.num_calibration_samples}-alpha{alpha}"
else:
    model_output_dir = f"{output_dir}/{model_name}-{sp_name}-ultrachat{args_ds2.num_calibration_samples}-alpha{alpha}"
os.makedirs(model_output_dir, exist_ok=True)

# Save as dense model (then it can be used with BnB load_in_8bit)
model.save_pretrained(
    model_output_dir, skip_sparsity_compression_stats=False,
    save_compressed=False,
    disable_sparse_compression=True,
)
# Note: config.json still contains `quantization_config` section, let's remove it
config_path = f'{model_output_dir}/config.json'
backup_path = f'{model_output_dir}/config.orig.json'
if os.path.exists(config_path):
    os.rename(config_path, backup_path)
with open(backup_path, 'r') as file:
    config = json.load(file)
if 'quantization_config' in config:
    del config['quantization_config']
with open(config_path, 'w') as file:
    json.dump(config, file, indent=2)

tokenizer.save_pretrained(model_output_dir)

