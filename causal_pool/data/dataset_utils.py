import json
import os.path as osp
import random
from typing import Dict

import jsonlines

from datasets import Dataset, concatenate_datasets


def process_entry(entry):
    metadata = entry["metadata"]
    question_type = metadata["question_type"]

    entry.pop("metadata")
    entry["question_type"] = question_type
    return entry

def load_causal_pool_dataset(dataset_name, random_seed=42, eval_size=128):
    # train with only counterfactual
    # eval with all question types
    dataset_base_path = osp.join("datasets", dataset_name)
    raw_train = list(
        jsonlines.open(
            osp.join(dataset_base_path, "splits", "train-counterfactual_velocity.jsonl")
        )
    ) + list(
        jsonlines.open(
            osp.join(dataset_base_path, "splits", "train-counterfactual_position.jsonl")
        )
    )
    raw_train = [process_entry(entry) for entry in raw_train]
    random.seed(random_seed)
    random.shuffle(raw_train)

    train_dataset = Dataset.from_list(raw_train[:-eval_size])
    eval_dataset = Dataset.from_list(raw_train[-eval_size:])

    # load descriptive
    raw_descriptive = list(jsonlines.open(osp.join(dataset_base_path, "splits", "train-descriptive.jsonl")))
    raw_descriptive = [process_entry(entry) for entry in raw_descriptive]
    random.shuffle(raw_descriptive)
    eval_descriptive = Dataset.from_list(raw_descriptive[:eval_size])
    eval_dataset = concatenate_datasets([eval_dataset, eval_descriptive])

    return train_dataset, eval_dataset


def gather_test_dataset(dataset_name, sizes: Dict[str, int], random_seed=42) -> Dataset:
    dataset_base_path = osp.join("datasets", dataset_name, "splits")
    names = [
        "counterfactual_velocity",
        "counterfactual_position",
        "descriptive",
        "predictive",
    ]
    datasets = []
    for name in sizes:
        if name not in names:
            raise ValueError(f"Invalid dataset name: {name}")
        size = sizes[name]
        raw = list(jsonlines.open(osp.join(dataset_base_path, "test-" + name + ".jsonl")))
        dataset = Dataset.from_list(raw).shuffle(seed=random_seed)
        # If size is -1, use all entries; otherwise select up to size
        if size != -1:
            dataset = dataset.select(range(size))
        datasets.append(dataset)
    return concatenate_datasets(datasets)
