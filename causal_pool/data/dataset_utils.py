import json
import os.path as osp
import random

import jsonlines

from datasets import Dataset


def load_causal_pool_dataset(dataset_name, random_seed=42, eval_size=128):
    # train with only counterfactual
    # eval with all question types
    dataset_base_path = osp.join("datasets", dataset_name)
    raw_train = list(
        jsonlines.open(osp.join(dataset_base_path, "splits", "counterfactual_train.jsonl"))
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
