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


def load_train_subset(dataset_name, subset_name, random_seed=42, eval_size=128):
    dataset_base_path = osp.join("datasets", dataset_name)
    raw_train = list(
        jsonlines.open(
            osp.join(dataset_base_path, "splits", "train-" + subset_name + ".jsonl")
        )
    )
    raw_train = [process_entry(entry) for entry in raw_train]
    random.seed(random_seed)
    random.shuffle(raw_train)
    train_dataset = Dataset.from_list(raw_train[:-eval_size])
    eval_dataset = Dataset.from_list(raw_train[-eval_size:])
    return train_dataset, eval_dataset


def load_counterfactual_train_dataset(dataset_name, random_seed=42, eval_size=128):
    counterfactual_velocity_train, counterfactual_velocity_eval = load_train_subset(
        dataset_name, "counterfactual_velocity", random_seed, eval_size
    )
    counterfactual_position_train, counterfactual_position_eval = load_train_subset(
        dataset_name, "counterfactual_position", random_seed, eval_size
    )
    descriptive_train, descriptive_eval = load_train_subset(
        dataset_name, "descriptive", random_seed, eval_size
    )

    # do NOT train on descriptive
    train_dataset = concatenate_datasets(
        [
            # counterfactual_velocity_train,
            counterfactual_position_train,
        ]
    ).shuffle(seed=random_seed)

    eval_dataset = concatenate_datasets(
        [
            counterfactual_velocity_eval,
            counterfactual_position_eval,
            descriptive_eval,
        ]
    )
    return train_dataset, eval_dataset


def load_descriptive_train_dataset(dataset_name, random_seed=42, eval_size=128):
    counterfactual_velocity_train, counterfactual_velocity_eval = load_train_subset(
        dataset_name, "counterfactual_velocity", random_seed, eval_size
    )
    counterfactual_position_train, counterfactual_position_eval = load_train_subset(
        dataset_name, "counterfactual_position", random_seed, eval_size
    )
    descriptive_train, descriptive_eval = load_train_subset(
        dataset_name, "descriptive", random_seed, eval_size
    )
    
    train_dataset = descriptive_train
    
    eval_dataset = concatenate_datasets(
        [
            counterfactual_velocity_eval,
            counterfactual_position_eval,
            descriptive_eval,
        ]
    )
    
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
        raw = list(
            jsonlines.open(osp.join(dataset_base_path, "test-" + name + ".jsonl"))
        )
        dataset = Dataset.from_list(raw).shuffle(seed=random_seed)
        # If size is -1, use all entries; otherwise select up to size
        if size != -1:
            dataset = dataset.select(range(size))
        datasets.append(dataset)
    return concatenate_datasets(datasets)
