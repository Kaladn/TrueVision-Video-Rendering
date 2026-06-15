import json
from pathlib import Path

from truevision_runtime.research_dataset_intake import (
    choose_dataset_targets,
    extract_hf_dataset_id,
    train_relevance_model,
)


def test_extract_hf_dataset_id_from_catalog_link():
    links = [
        "Paper: https://example.test/paper",
        "Dataset: https://huggingface.co/datasets/owner/name-with-dash",
    ]

    assert extract_hf_dataset_id(links) == "owner/name-with-dash"


def test_choose_dataset_targets_prefers_high_priority_dataset_links(tmp_path: Path):
    catalog = tmp_path / "catalog.jsonl"
    rows = [
        {
            "item_type": "paper",
            "title": "Paper With Dataset Link",
            "priority": "high",
            "tags": ["motion_address"],
            "truevision_use": "reference",
            "links": ["Dataset: https://huggingface.co/datasets/no/paper"],
        },
        {
            "item_type": "dataset",
            "title": "Low Dataset",
            "priority": "low",
            "tags": [],
            "truevision_use": "background",
            "links": ["Dataset: https://huggingface.co/datasets/no/low"],
        },
        {
            "item_type": "benchmark",
            "title": "High Benchmark",
            "priority": "high",
            "tags": ["physics_state"],
            "truevision_use": "state verification",
            "links": ["Dataset: https://huggingface.co/datasets/yes/high"],
        },
    ]
    catalog.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    targets = choose_dataset_targets(catalog, limit=2)

    assert [target.dataset_id for target in targets] == ["yes/high", "no/low"]


def test_train_relevance_model_scores_crawled_motion_dataset():
    catalog_rows = [
        {
            "title": "Motion trajectory reward benchmark",
            "priority": "high",
            "tags": ["motion_address", "post_training_alignment"],
            "truevision_use": "motion-address control",
            "links": [],
        },
        {
            "title": "Generic archive",
            "priority": "low",
            "tags": [],
            "truevision_use": "background reference",
            "links": [],
        },
    ]
    crawled_rows = [
        {
            "dataset_id": "demo/motion",
            "title": "Motion trajectory dataset",
            "tags": ["motion_address"],
            "truevision_use": "motion-address control",
        }
    ]

    model = train_relevance_model(catalog_rows, crawled_rows)

    assert model["training_rows"] == 2
    assert model["positive_rows"] == 1
    assert model["feature_count"] > 0
    assert model["crawled_dataset_scores"][0]["dataset_id"] == "demo/motion"
