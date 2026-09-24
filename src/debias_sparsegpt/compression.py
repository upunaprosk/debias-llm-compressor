"""Compression orchestration for Debias-SparseGPT."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from llmcompressor.entrypoints.oneshot import Oneshot


@dataclass(frozen=True)
class CompressionConfig:
    model: str
    recipe: Path
    output_dir: Path

    sparsity: str = "2:4"
    alpha: float = 0.0
    seed: int = 1

    preprocessing_num_workers: int = 4

    torch_dtype: torch.dtype = torch.bfloat16


def seed_everything(seed: int) -> None:
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def load_model_and_tokenizer(
    config: CompressionConfig,
):
    """Load the model."""

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=config.torch_dtype,
    )

    # Preserved compatibility workaround from the source experiment.
    model.generation_config.do_sample = True

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    return model, tokenizer


def create_oneshot_session(
    *,
    model,
    bootstrap_dataset,
    config: CompressionConfig,
) -> "Oneshot":
    """Create the llm-compressor Oneshot session."""

    from llmcompressor.entrypoints.oneshot import Oneshot

    seed_everything(config.seed)

    os.environ["ALPHA"] = str(config.alpha)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return Oneshot(
        model=model,
        dataset=bootstrap_dataset,
        recipe=str(config.recipe),
        preprocessing_num_workers=config.preprocessing_num_workers,
        num_calibration_samples=5,
        max_seq_length=100,
    )


def apply_prepared_calibration(
    *,
    session: "Oneshot",
    calibration_dataloader,
) -> None:
    session.apply_recipe_modifiers(calibration_dataloader=calibration_dataloader)


def build_output_name(
    *,
    model_name: str,
    sparsity: str,
    stereoset_samples: int,
    ultrachat_samples: int,
    alpha: float | str,
) -> str:
    """
    Build model-output following source naming convention.
    """

    sparse_name = "sparse" + sparsity.replace(":", "")

    if stereoset_samples > 0 and ultrachat_samples > 0:
        return (
            f"{model_name}-{sparse_name}"
            f"-stereo{stereoset_samples}"
            f"-ultrachat{ultrachat_samples}"
            f"-alpha{alpha}"
        )

    if stereoset_samples > 0:
        return f"{model_name}-{sparse_name}-stereo{stereoset_samples}-alpha{alpha}"

    return f"{model_name}-{sparse_name}-ultrachat{ultrachat_samples}-alpha{alpha}"


def remove_quantization_config(
    model_directory: Path,
) -> None:
    """
    The generated config.json is backed up to config.orig.json and
    quantization_config is removed from the active config.
    """

    config_path = model_directory / "config.json"

    backup_path = model_directory / "config.orig.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {config_path}")

    if backup_path.exists():
        backup_path.unlink()

    config_path.rename(backup_path)

    with backup_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    config.pop(
        "quantization_config",
        None,
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )


def save_dense_model(
    *,
    model,
    tokenizer,
    base_output_dir: Path,
    model_name: str,
    sparsity: str,
    stereoset_samples: int,
    ultrachat_samples: int,
    alpha: float,
) -> Path:
    model.generation_config.do_sample = True

    directory_name = build_output_name(
        model_name=model_name,
        sparsity=sparsity,
        stereoset_samples=stereoset_samples,
        ultrachat_samples=ultrachat_samples,
        alpha=alpha,
    )

    output_path = Path(base_output_dir) / directory_name

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_path,
        skip_sparsity_compression_stats=False,
        save_compressed=False,
        disable_sparse_compression=True,
    )

    remove_quantization_config(output_path)

    tokenizer.save_pretrained(output_path)

    return output_path
