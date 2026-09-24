"""Command-line interface for Debias-SparseGPT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from debias_sparsegpt.calibration.stereoset import (
    build_stereoset_dataset,
)
from debias_sparsegpt.calibration.ultrachat import (
    MixedCalibrationConfig,
    prepare_mixed_calibration,
)
from debias_sparsegpt.compression import (
    CompressionConfig,
    apply_prepared_calibration,
    create_oneshot_session,
    load_model_and_tokenizer,
    save_dense_model,
)


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def default_alpha() -> float:
    """Read ALPHA from the env variables."""

    return float(os.environ.get("ALPHA", "0"))


def add_shared_arguments(
    parser: argparse.ArgumentParser,
) -> None:

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model identifier or local model path.",
    )

    parser.add_argument(
        "--recipe",
        required=True,
        type=Path,
        help="Path to the llm-compressor sparsity recipe.",
    )

    parser.add_argument(
        "--sparsity",
        choices=("1:4", "2:4"),
        default="2:4",
        help="Structured sparsity pattern.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=default_alpha(),
        help=(
            "Debias-SparseGPT coefficient. "
            "Defaults to the ALPHA environment variable or 0."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of preprocessing workers.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Debias-SparseGPT command-line parser."""

    parser = argparse.ArgumentParser(
        prog="debias-sparsegpt",
        description=(
            "Reproduce Debias-SparseGPT pruning experiments."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    stereoset = subparsers.add_parser(
        "stereoset",
        help="Run the StereoSet-only calibration experiment.",
    )

    add_shared_arguments(stereoset)

    stereoset.add_argument(
        "--stereoset",
        type=Path,
        default=None,
        help=(
            "Optional path to StereoSet dev.json. "
            "If omitted, use the cached file or original remote source."
        ),
    )

    stereoset.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_llama8b_2of4"),
        help="Output directory.",
    )

    stereoset.add_argument(
        "--max-seq-length",
        type=int,
        default=100,
        help=(
            "Maximum calibration sequence length. "
            "The original StereoSet experiment used 100."
        ),
    )

    ultrachat = subparsers.add_parser(
        "ultrachat",
        help=(
            "Run the mixed StereoSet + UltraChat calibration experiment."
        ),
    )

    add_shared_arguments(ultrachat)

    ultrachat.add_argument(
        "--stereoset",
        type=Path,
        default=None,
        help=(
            "Optional path to StereoSet dev.json. "
            "If omitted, use the cached file or original remote source."
        ),
    )

    ultrachat.add_argument(
        "--stereoset-samples",
        type=int,
        default=None,
        help=(
            "Number of StereoSet calibration samples. "
            "Defaults to the complete extracted StereoSet set."
        ),
    )

    ultrachat.add_argument(
        "--ultrachat-samples",
        type=int,
        default=256,
        help=(
            "Number of UltraChat calibration samples. "
            "The original implementation used 256."
        ),
    )

    ultrachat.add_argument(
        "--stereoset-max-seq-length",
        type=int,
        default=64,
        help=(
            "Maximum sequence length for StereoSet in the mixed setup."
        ),
    )

    ultrachat.add_argument(
        "--ultrachat-max-seq-length",
        type=int,
        default=1024,
        help=(
            "Maximum sequence length for UltraChat calibration."
        ),
    )

    ultrachat.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_models"),
        help="Base directory for generated model checkpoints.",
    )

    return parser


def make_config(
    args: argparse.Namespace,
) -> CompressionConfig:
    """Create the shared compression configuration."""

    return CompressionConfig(
        model=args.model,
        recipe=args.recipe,
        output_dir=args.output_dir,
        sparsity=args.sparsity,
        alpha=args.alpha,
        seed=args.seed,
        preprocessing_num_workers=args.workers,
    )


def run_stereoset(
    args: argparse.Namespace,
) -> None:
    """
    Run the StereoSet-only experiment.

    This preserves the original flat StereoSet calibration dataset.
    """

    from llmcompressor import oneshot

    config = make_config(args)

    dataset = build_stereoset_dataset(
        source=args.stereoset,
    )

    model, tokenizer = load_model_and_tokenizer(
        config
    )

    os.environ["ALPHA"] = str(config.alpha)

    print(f"Calibration samples: {len(dataset)}")
    print(f"Debias alpha = {config.alpha}")

    sparse_model = oneshot(
        model=model,
        dataset=dataset,
        recipe=str(config.recipe),
        num_calibration_samples=len(dataset),
        preprocessing_num_workers=(
            config.preprocessing_num_workers
        ),
        max_seq_length=args.max_seq_length,
        stage="sparsity_stage",
        shuffle_calibration_samples=False,
    )

    output_path = (
        config.output_dir
        / "sparsity_stage"
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    sparse_model.save_pretrained(
        output_path,
        skip_sparsity_compression_stats=False,
        save_compressed=True,
        disable_sparse_compression=False,
    )

    tokenizer.save_pretrained(
        output_path
    )

    print(f"Saved model to {output_path}")


def run_ultrachat(
    args: argparse.Namespace,
) -> None:
    """
    Run the original mixed StereoSet + UltraChat experiment.
    """

    config = make_config(args)

    stereoset_dataset = build_stereoset_dataset(
        source=args.stereoset,
    )

    model, tokenizer = load_model_and_tokenizer(
        config
    )

    bootstrap_loader = DataLoader(
        stereoset_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=None,
    )

    session = create_oneshot_session(
        model=model,
        bootstrap_dataset=bootstrap_loader,
        config=config,
    )

    calibration_config = MixedCalibrationConfig(
        stereoset_samples=args.stereoset_samples,
        ultrachat_samples=args.ultrachat_samples,
        stereoset_batch_size=2,
        ultrachat_batch_size=1,
        stereoset_max_seq_length=(
            args.stereoset_max_seq_length
        ),
        ultrachat_max_seq_length=(
            args.ultrachat_max_seq_length
        ),
    )

    prepared = prepare_mixed_calibration(
        oneshot_instance=session,
        stereoset_dataset=stereoset_dataset,
        config=calibration_config,
    )

    print("Dataset(s):")

    if prepared.stereoset_samples > 0:
        print(
            "StereoSet "
            f"length={prepared.stereoset_samples} "
            f"batch_size={calibration_config.stereoset_batch_size} "
            f"max_seq_length="
            f"{calibration_config.stereoset_max_seq_length}"
        )

    if prepared.ultrachat_samples > 0:
        print(
            "UltraChat "
            f"length={prepared.ultrachat_samples} "
            f"batch_size={calibration_config.ultrachat_batch_size} "
            f"max_seq_length="
            f"{calibration_config.ultrachat_max_seq_length}"
        )

    print(f"Debias alpha = {config.alpha}")

    apply_prepared_calibration(
        session=session,
        calibration_dataloader=prepared.dataloader,
    )

    model_name = (
        args.model.rstrip("/")
        .split("/")[-1]
    )

    output_path = save_dense_model(
        model=model,
        tokenizer=tokenizer,
        base_output_dir=config.output_dir,
        model_name=model_name,
        sparsity=config.sparsity,
        stereoset_samples=prepared.stereoset_samples,
        ultrachat_samples=prepared.ultrachat_samples,
        alpha=config.alpha,
    )

    print(f"Saved model to {output_path}")


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stereoset":
        run_stereoset(args)
        return

    if args.command == "ultrachat":
        run_ultrachat(args)
        return

    parser.error(
        f"Unknown command: {args.command}"
    )


if __name__ == "__main__":
    main()