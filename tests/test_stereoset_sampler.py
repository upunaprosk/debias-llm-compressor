"""Tests for pair ordering in mixed StereoSet calibration.

Run with the repository's patched llm-compressor 0.8.1 installed.
No tokenizer, model, or GPU is required.
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
from datasets import Dataset
from torch.utils.data import SequentialSampler

from debias_sparsegpt.calibration.ultrachat import (
    MixedCalibrationConfig,
    prepare_mixed_calibration,
)
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets.utils import (
    LengthAwareSampler,
    format_calibration_data,
)


def _stereoset():
    # Adjacent rows belong to one X0/X1 pair, but their lengths differ.
    # Sorting by length will therefore separate the pairs.
    lengths = [3, 13, 12, 4, 6, 15, 14, 5]
    return Dataset.from_list(
        [
            {
                "input_ids": [100 + i] * length,
                "attention_mask": [1] * length,
                "row_id": i,
                "pair_id": i // 2,
            }
            for i, length in enumerate(lengths)
        ]
    )


def _args(dataset, *, samples=6, shuffle=True, no_sampler=None):
    args = SimpleNamespace(
        dataset=dataset,
        splits=None,
        batch_size=2,
        num_calibration_samples=samples,
        shuffle_calibration_samples=shuffle,
        max_seq_length=64,
        # Return original rows: no tokenizer/collator padding is needed here.
        data_collator=lambda rows: rows,
    )
    if no_sampler is not None:
        args.no_sampler = no_sampler
    return args


def _rows(loader):
    return [row for batch in loader for row in batch]


def test_no_sampler_is_declared():
    assert "no_sampler" in {field.name for field in fields(DatasetArguments)}


def test_no_sampler_preserves_pairs_and_sample_limit():
    dataset = _stereoset()
    args = _args(dataset, samples=6, shuffle=False, no_sampler=True)
    loader = format_calibration_data(args, dataset, processor=None)

    assert isinstance(loader.sampler, SequentialSampler)
    assert list(loader.sampler) == list(range(6))
    assert len(loader) == 3

    rows = _rows(loader)
    assert [row["row_id"] for row in rows] == list(range(6))
    assert [row["pair_id"] for row in rows] == [0, 0, 1, 1, 2, 2]
    assert [[row["pair_id"] for row in batch] for batch in loader] == [
        [0, 0],
        [1, 1],
        [2, 2],
    ]


def test_length_aware_sampler_can_break_pairs_without_override():
    dataset = _stereoset()
    args = _args(dataset, samples=len(dataset), shuffle=False, no_sampler=False)
    loader = format_calibration_data(args, dataset, processor=None)

    assert isinstance(loader.sampler, LengthAwareSampler)
    assert list(loader.sampler) != list(range(len(dataset)))
    pairs = [row["pair_id"] for row in _rows(loader)]
    assert any(pairs[i] != pairs[i + 1] for i in range(0, len(pairs), 2))


def test_mixed_pipeline_preserves_stereoset_order():
    dataset = _stereoset()
    base_args = _args(dataset, samples=8, shuffle=True)
    instance = SimpleNamespace(dataset_args=base_args, processor=None)

    result = prepare_mixed_calibration(
        oneshot_instance=instance,
        stereoset_dataset=dataset,
        config=MixedCalibrationConfig(stereoset_samples=6, ultrachat_samples=0),
    )

    loader = result.dataloader
    assert result.stereoset_samples == 6
    assert result.ultrachat_samples == 0
    assert isinstance(loader.sampler, SequentialSampler)
    assert list(loader.sampler) == list(range(6))
    assert loader.batch_size == 2
    assert [row["row_id"] for row in _rows(loader)] == list(range(6))
    assert [[row["pair_id"] for row in batch] for batch in loader] == [
        [0, 0],
        [1, 1],
        [2, 2],
    ]
    # StereoSet's override must not change the shared arguments for UltraChat.
    assert base_args.shuffle_calibration_samples is True


@pytest.mark.parametrize(
    ("samples", "batch_size", "message"),
    [
        (5, 2, "even"),
        (6, 4, "batch_size must be 2"),
    ],
)
def test_mixed_rejects_invalid_pair_configuration(samples, batch_size, message):
    dataset = _stereoset()
    instance = SimpleNamespace(
        dataset_args=_args(dataset),
        processor=None,
    )
    config = MixedCalibrationConfig(
        stereoset_samples=samples,
        ultrachat_samples=0,
        stereoset_batch_size=batch_size,
    )

    with pytest.raises(ValueError, match=message):
        prepare_mixed_calibration(
            oneshot_instance=instance,
            stereoset_dataset=dataset,
            config=config,
        )
