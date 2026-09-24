from pathlib import Path

from debias_sparsegpt.cli import (
    build_parser,
    make_config,
)


def test_stereoset_cli():
    parser = build_parser()

    args = parser.parse_args(
        [
            "stereoset",
            "--recipe",
            "recipes/structured_2of4.yaml",
        ]
    )

    assert args.command == "stereoset"
    assert args.sparsity == "2:4"
    assert args.alpha == 0.0
    assert args.seed == 1
    assert args.max_seq_length == 100


def test_stereoset_custom_arguments():
    parser = build_parser()

    args = parser.parse_args(
        [
            "stereoset",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--recipe",
            "recipes/structured_1of4.yaml",
            "--sparsity",
            "1:4",
            "--alpha",
            "0.25",
            "--seed",
            "5",
            "--max-seq-length",
            "128",
        ]
    )

    assert args.model == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert args.sparsity == "1:4"
    assert args.alpha == 0.25
    assert args.seed == 5
    assert args.max_seq_length == 128


def test_mixed_cli():
    parser = build_parser()

    args = parser.parse_args(
        [
            "mixed",
            "--recipe",
            "recipes/structured_2of4.yaml",
            "--stereoset-samples",
            "500",
            "--ultrachat-samples",
            "256",
        ]
    )

    assert args.command == "mixed"

    assert args.stereoset_samples == 500
    assert args.ultrachat_samples == 256

    assert args.stereoset_max_seq_length == 64
    assert args.ultrachat_max_seq_length == 1024


def test_make_config():
    parser = build_parser()

    args = parser.parse_args(
        [
            "stereoset",
            "--model",
            "test-model",
            "--recipe",
            "recipes/test.yaml",
            "--output-dir",
            "output",
            "--alpha",
            "0.1",
            "--seed",
            "3",
        ]
    )

    config = make_config(args)

    assert config.model == "test-model"
    assert config.recipe == Path(
        "recipes/test.yaml"
    )
    assert config.output_dir == Path(
        "output"
    )

    assert config.alpha == 0.1
    assert config.seed == 3
    assert config.sparsity == "2:4"