import json
import os.path as osp
import random
from typing import Dict

import jsonlines

from datasets import Dataset, concatenate_datasets


def load_causal_pool_dataset(dataset_name, random_seed=42, eval_size=128):
    # train with only counterfactual
    # eval with all question types
    dataset_base_path = osp.join("datasets", dataset_name)
    raw_train = list(
        jsonlines.open(
            osp.join(dataset_base_path, "splits", "counterfactual_train.jsonl")
        )
    )

    new_raw = []
    bad_videos = set(
        e["video"]
        for e in json.load(open(osp.join(dataset_base_path, "bad_videos.json")))[
            "bad_videos"
        ]
    )
    for entry in raw_train:
        if entry["video"] not in bad_videos:
            new_raw.append(entry)
    raw_train = new_raw

    random.seed(random_seed)
    random.shuffle(raw_train)

    train_dataset = Dataset.from_list(raw_train[:-eval_size])
    eval_dataset = Dataset.from_list(raw_train[-eval_size:])

    return train_dataset, eval_dataset


def gather_test_dataset(dataset_name, sizes: Dict[str, int], random_seed=42) -> Dataset:
    dataset_base_path = osp.join("datasets", dataset_name, "splits")
    names = [
        "counterfactual_test",
        "descriptive",
        "predictive",
    ]
    datasets = []
    for name in sizes:
        if name not in names:
            raise ValueError(f"Invalid dataset name: {name}")
        size = sizes[name]
        raw = list(jsonlines.open(osp.join(dataset_base_path, name + ".jsonl")))
        dataset = Dataset.from_list(raw).shuffle(seed=random_seed).select(range(size))
        datasets.append(dataset)
    return concatenate_datasets(datasets)
