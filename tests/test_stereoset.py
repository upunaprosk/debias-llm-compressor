from debias_sparsegpt.calibration.stereoset import (
    build_stereoset_dataset,
    extract_intrasentence_examples,
)


def make_stereoset_data():
    return {
        "data": {
            "intrasentence": [
                {
                    "sentences": [
                        {
                            "sentence": "Stereotype sentence.",
                            "gold_label": "stereotype",
                        },
                        {
                            "sentence": "Unrelated sentence.",
                            "gold_label": "unrelated",
                        },
                        {
                            "sentence": "Anti-stereotype sentence.",
                            "gold_label": "anti-stereotype",
                        },
                    ]
                },
                {
                    "sentences": [
                        {
                            "sentence": "Another stereotype.",
                            "gold_label": "stereotype",
                        },
                        {
                            "sentence": "Another anti-stereotype.",
                            "gold_label": "anti-stereotype",
                        },
                    ]
                },
            ]
        }
    }


def test_extract_intrasentence_examples():
    data = make_stereoset_data()

    examples = extract_intrasentence_examples(data)

    assert examples == [
        "Stereotype sentence.",
        "Anti-stereotype sentence.",
        "Another stereotype.",
        "Another anti-stereotype.",
    ]


def test_preserves_source_order():
    data = {
        "data": {
            "intrasentence": [
                {
                    "sentences": [
                        {
                            "sentence": "Anti first.",
                            "gold_label": "anti-stereotype",
                        },
                        {
                            "sentence": "Stereo second.",
                            "gold_label": "stereotype",
                        },
                    ]
                }
            ]
        }
    }

    examples = extract_intrasentence_examples(data)

    assert examples == [
        "Anti first.",
        "Stereo second.",
    ]


def test_skips_incomplete_pairs():
    data = {
        "data": {
            "intrasentence": [
                {
                    "sentences": [
                        {
                            "sentence": "Only stereotype.",
                            "gold_label": "stereotype",
                        },
                        {
                            "sentence": "Unrelated.",
                            "gold_label": "unrelated",
                        },
                    ]
                }
            ]
        }
    }

    examples = extract_intrasentence_examples(data)

    assert examples == []


def test_ignores_unrelated_examples():
    data = make_stereoset_data()

    examples = extract_intrasentence_examples(data)

    assert "Unrelated sentence." not in examples


def test_build_stereoset_dataset(monkeypatch):
    data = make_stereoset_data()

    monkeypatch.setattr(
        "debias_sparsegpt.calibration.stereoset.load_stereoset",
        lambda source=None: data,
    )

    dataset = build_stereoset_dataset()

    assert len(dataset) == 4
    assert dataset.column_names == ["text"]
    assert dataset[0]["text"] == "Stereotype sentence."
    assert dataset[1]["text"] == "Anti-stereotype sentence."


def original_extraction(data):
    examples = []

    for entry in data["data"]["intrasentence"]:
        check_set = set()

        for sentence_entry in entry["sentences"]:
            gold_label = sentence_entry["gold_label"]

            if gold_label in [
                "stereotype",
                "anti-stereotype",
            ]:
                check_set.add(gold_label)

        if len(check_set) < 2:
            continue

        for sentence_entry in entry["sentences"]:
            if sentence_entry["gold_label"] in check_set:
                examples.append(sentence_entry["sentence"])

    return examples


def test_refactor_matches_original_extraction():
    data = make_stereoset_data()

    expected = original_extraction(data)

    actual = extract_intrasentence_examples(data)

    assert actual == expected
