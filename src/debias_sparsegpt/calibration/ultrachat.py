"""Mixed StereoSet + UltraChat calibration for Debias-SparseGPT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable


from torch.utils.data import DataLoader, SequentialSampler
class CombinedDataLoader:
    """
    Iterate through multiple calibration dataloaders sequentially.
    """

    def __init__(self, dataloaders: Iterable[DataLoader]) -> None:
        self.dataloaders = list(dataloaders)

        try:
            self._length = sum(len(loader) for loader in self.dataloaders)
        except TypeError:
            self._length = None

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("Length is unavailable for one or more dataloaders.")

        return self._length


@dataclass(frozen=True)
class MixedCalibrationConfig:
    stereoset_samples: int | None = None
    ultrachat_samples: int = 256

    stereoset_batch_size: int = 2
    ultrachat_batch_size: int = 1

    stereoset_max_seq_length: int = 64
    ultrachat_max_seq_length: int = 1024

    ultrachat_split: str = "train_gen[:1%]"

    def validate(self) -> None:
        if self.stereoset_samples is not None and self.stereoset_samples < 0:
            raise ValueError("stereoset_samples must be non-negative.")

        if self.ultrachat_samples < 0:
            raise ValueError("ultrachat_samples must be non-negative.")

        if (self.stereoset_samples or 0) == 0 and self.ultrachat_samples == 0:
            raise ValueError("At least one calibration dataset must contain samples.")


@dataclass
class PreparedCalibration:
    """Prepared dataloader with the effective sample counts."""

    dataloader: object
    stereoset_samples: int
    ultrachat_samples: int


def prepare_mixed_calibration(
    *,
    oneshot_instance,
    stereoset_dataset,
    config: MixedCalibrationConfig,
) -> PreparedCalibration:
    """
    StereoSet + UltraChat calibration setup using llm-compressor's DatasetArguments and
    get_calibration_dataloader API.
    """

    from llmcompressor.datasets.utils import (
        get_calibration_dataloader,
    )

    config.validate()

    available_stereoset = len(stereoset_dataset)

    requested_stereoset = (
        available_stereoset if config.stereoset_samples is None else config.stereoset_samples
    )

    stereoset_samples = min(
        available_stereoset,
        requested_stereoset,
    )

    ultrachat_samples = min(
        2000,
        config.ultrachat_samples,
    )

    loaders = []


    if stereoset_samples > 0:

        # StereoSet must contain complete X0/X1 pairs.
        if config.stereoset_batch_size != 2:
            raise ValueError(
                "StereoSet batch_size must be 2."
            )

        if stereoset_samples % 2 != 0:
            raise ValueError(
                "StereoSet sample count must be even."
            )

        stereoset_args = deepcopy(oneshot_instance.dataset_args)

        stereoset_args.splits = None
        stereoset_args.dataset = stereoset_dataset
        stereoset_args.batch_size = config.stereoset_batch_size
        stereoset_args.num_calibration_samples = stereoset_samples
        stereoset_args.max_seq_length = config.stereoset_max_seq_length

        # Preserve the original X0/X1 ordering.
        stereoset_args.shuffle_calibration_samples = False
        stereoset_args.no_sampler = True

        stereoset_loader = get_calibration_dataloader(
            stereoset_args,
            processor=oneshot_instance.processor,
        )

        # Runtime verification.
        assert isinstance(
            stereoset_loader.sampler,
            SequentialSampler,
        ), (
            "Unexpected StereoSet sampler: "
            f"{type(stereoset_loader.sampler).__name__}"
        )

        actual_indices = list(stereoset_loader.sampler)
        expected_indices = list(range(stereoset_samples))

        assert actual_indices == expected_indices, (
            "StereoSet order corrupted!"
        )

        print(
            "[StereoSet OK (NOT SHUFFLED)]",
            f"sampler={type(stereoset_loader.sampler).__name__}",
            f"samples={len(actual_indices)}",
            f"batch_size={stereoset_loader.batch_size}",
        )

        loaders.append(stereoset_loader)

    if ultrachat_samples > 0:
        ultrachat_args = deepcopy(oneshot_instance.dataset_args)

        ultrachat_args.splits = {"calibration": config.ultrachat_split}
        ultrachat_args.dataset = "ultrachat-200k"
        ultrachat_args.batch_size = config.ultrachat_batch_size
        ultrachat_args.num_calibration_samples = ultrachat_samples
        ultrachat_args.max_seq_length = config.ultrachat_max_seq_length

        ultrachat_loader = get_calibration_dataloader(
            ultrachat_args,
            processor=oneshot_instance.processor,
        )

        loaders.append(ultrachat_loader)

    if not loaders:
        raise ValueError("No calibration dataloaders were created.")

    if len(loaders) == 1:
        combined = loaders[0]
    else:
        combined = CombinedDataLoader(loaders)

    return PreparedCalibration(
        dataloader=combined,
        stereoset_samples=stereoset_samples,
        ultrachat_samples=ultrachat_samples,
    )
