import json

import pytest

from debias_sparsegpt.compression import (
    build_output_name,
    remove_quantization_config,
)


def test_build_mixed_output_name():
    name = build_output_name(
        model_name="Llama-3.1-8B-Instruct",
        sparsity="2:4",
        stereoset_samples=1000,
        ultrachat_samples=256,
        alpha=0.1,
    )

    assert name == (
        "Llama-3.1-8B-Instruct-"
        "sparse24-"
        "stereo1000-"
        "ultrachat256-"
        "alpha0.1"
    )


def test_build_stereoset_only_output_name():
    name = build_output_name(
        model_name="Llama-3.1-8B-Instruct",
        sparsity="1:4",
        stereoset_samples=500,
        ultrachat_samples=0,
        alpha=0.2,
    )

    assert name == (
        "Llama-3.1-8B-Instruct-"
        "sparse14-"
        "stereo500-"
        "alpha0.2"
    )


def test_build_ultrachat_only_output_name():
    name = build_output_name(
        model_name="Llama-3.1-8B-Instruct",
        sparsity="2:4",
        stereoset_samples=0,
        ultrachat_samples=256,
        alpha=0.0,
    )

    assert name == (
        "Llama-3.1-8B-Instruct-"
        "sparse24-"
        "ultrachat256-"
        "alpha0.0"
    )


def test_remove_quantization_config(tmp_path):
    config_path = tmp_path / "config.json"

    config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "quantization_config": {
            "some_setting": True,
        },
    }

    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    remove_quantization_config(tmp_path)

    updated = json.loads(
        config_path.read_text(
            encoding="utf-8",
        )
    )

    backup = json.loads(
        (tmp_path / "config.orig.json").read_text(
            encoding="utf-8",
        )
    )

    assert "quantization_config" not in updated
    assert updated["model_type"] == "llama"

    assert "quantization_config" in backup


def test_remove_quantization_config_without_quantization(tmp_path):
    config_path = tmp_path / "config.json"

    config = {
        "model_type": "llama",
    }

    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    remove_quantization_config(tmp_path)

    updated = json.loads(
        config_path.read_text(
            encoding="utf-8",
        )
    )

    assert updated == {
        "model_type": "llama",
    }


def test_remove_quantization_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        remove_quantization_config(tmp_path)