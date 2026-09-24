"""StereoSet calibration data used by Debias-SparseGPT."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from datasets import Dataset


DEFAULT_STEREOSET_URL = (
    "https://raw.githubusercontent.com/"
    "gsgoncalves/EMNLP2023_llm_compression_and_social_bias/"
    "refs/heads/main/data/stereoset/dev.json"
)


def resolve_stereoset_source(
    source: str | Path | None = None,
) -> str:
    """
    Resolve the StereoSet source used by the source experiments.
    """

    if source is not None:
        return str(source)

    cached = (
        Path.home()
        / ".cache"
        / "stereoset"
        / "dev.json"
    )

    if cached.exists():
        return str(cached)

    return DEFAULT_STEREOSET_URL


def load_stereoset(
    source: str | Path | None = None,
) -> dict:
    """Load the StereoSet JSON"""

    resolved = resolve_stereoset_source(source)

    if resolved.startswith(("http://", "https://")):
        with urllib.request.urlopen(resolved) as response:
            return json.load(response)

    with Path(resolved).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_intrasentence_examples(
    data: dict,
) -> list[str]:
    """
    Extract StereoSet intrasentence calibration examples.
    """

    try:
        entries = data["data"]["intrasentence"]
    except KeyError as exc:
        raise ValueError(
            "Invalid StereoSet data: expected "
            "data['data']['intrasentence']."
        ) from exc

    examples: list[str] = []

    valid_labels = {
        "stereotype",
        "anti-stereotype",
    }

    for entry in entries:
        labels = {
            sentence["gold_label"]
            for sentence in entry["sentences"]
            if sentence["gold_label"] in valid_labels
        }

        if not valid_labels.issubset(labels):
            continue

        for sentence in entry["sentences"]:
            if sentence["gold_label"] in valid_labels:
                examples.append(sentence["sentence"])

    return examples


def build_stereoset_dataset(
    source: str | Path | None = None,
) -> Dataset:
    """
    Build the flat Hugging Face Dataset used for calibration.
    """

    data = load_stereoset(source)
    examples = extract_intrasentence_examples(data)

    if not examples:
        raise ValueError(
            "No valid StereoSet calibration examples were found."
        )

    return Dataset.from_dict(
        {
            "text": examples,
        }
    )