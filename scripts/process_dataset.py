import jsonlines
from collections import defaultdict
import random

raw = list(jsonlines.open("datasets/1k_simple/raw_qa.jsonl"))

types_to_entries = defaultdict(list)

for entry in raw:
    types_to_entries[entry["metadata"]["question_type"]].append(entry)

all_counterfactual_entries = []


for question_type, entries in types_to_entries.items():
    print(f"{question_type}: {len(entries)}")

    if "counterfactual" in question_type:
        all_counterfactual_entries.extend(entries)
    else:
        jsonlines.open(f"splits/{question_type}.jsonl", "w").write_all(entries)

random.seed(42)
# Shuffle counterfactual entries after collecting them to ensure proper randomization
random.shuffle(all_counterfactual_entries)

test_size = 512
train_counterfactual = all_counterfactual_entries[:-test_size]
test_counterfactual = all_counterfactual_entries[-test_size:]

jsonlines.open("splits/counterfactual_train.jsonl", "w").write_all(train_counterfactual)
jsonlines.open("splits/counterfactual_test.jsonl", "w").write_all(test_counterfactual)
