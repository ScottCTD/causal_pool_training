import os.path as osp
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from custom_trainer import CausalPoolTrainer
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    EvalPrediction,
    GenerationConfig,
    Qwen3VLForConditionalGeneration,
    Seq2SeqTrainingArguments,
)

from causal_pool.data.dataset_utils import load_causal_pool_dataset
from causal_pool.prompt_utils import build_question_prompt, index_to_letter

DATASET_NAME = "1k_simple"
train_dataset, eval_dataset = load_causal_pool_dataset(DATASET_NAME, eval_size=320)

model_name = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(
    model_name, local_files_only=True, use_fast=True
)
tokenizer = processor.tokenizer

peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=[
        # Text tower
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Vision tower (attention and MLP)
        "qkv",
        "proj",
        "linear_fc1",
        "linear_fc2",
    ],
)
model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters()


def get_assistant_mask(input_ids):
    # Vectorized search for sequence "<|im_start|>assistant\n" -> [151644, 77091, 198]
    pattern = torch.tensor(
        [151644, 77091, 198], device=input_ids.device, dtype=input_ids.dtype
    )
    k = pattern.numel()
    batch_size, seq_len = input_ids.shape

    if seq_len < k:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    # Create all sliding windows of length k: shape (B, T-k+1, k)
    windows = input_ids.unfold(1, k, 1)
    # Match windows against the pattern
    matches = (windows == pattern).all(dim=-1)  # (B, T-k+1)

    any_match = matches.any(dim=1)
    # First occurrence index (undefined if no match, so guard with any_match)
    first_pos = torch.where(
        any_match,
        matches.int().argmax(dim=1),
        torch.full((batch_size,), seq_len, device=input_ids.device, dtype=torch.long),
    )

    start_after = first_pos + k
    mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(
        0
    ) >= start_after.unsqueeze(1)
    return mask


def get_model_inputs(conversations, add_generation_prompt):
    text = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        padding=True,
    )
    images, videos, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    # split the videos and according metadatas
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # since qwen-vl-utils has resize the images/videos,
    # we should pass do_resize=False to avoid duplicate operation in processor!
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        truncation=False,
        max_length=None,
        padding=True,
        padding_side="left",
        **video_kwargs,
    )
    return inputs


def train_data_collator(samples):
    conversations = []
    for sample in samples:
        video_name = sample["video"]
        video_path = osp.join(
            "datasets", DATASET_NAME, "shots", video_name, "video.mp4"
        )

        question_prompt = build_question_prompt(sample)

        ground_truth_indices = sample["ground_truth"]  # List[int] indicies
        ground_truth_answer = "".join(index_to_letter(i) for i in ground_truth_indices)

        # TODO: further clarify min/max/total pixels and video tokens
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 4 * 32 * 32,
                        "max_pixels": 256 * 32 * 32,
                        "total_pixels": 20480 * 32 * 32,
                    },
                    {"type": "text", "text": question_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": ground_truth_answer,
            },
        ]
        conversations.append(conversation)

    inputs = get_model_inputs(conversations, add_generation_prompt=False)

    # construct labels
    input_ids = inputs.input_ids
    # TODO: there's a bug here, the assistant mask doesn't mask the turn end token
    assistant_mask = get_assistant_mask(input_ids)
    labels = inputs.input_ids.clone()
    labels[~assistant_mask] = -100

    # construct generation inputs
    # TODO: efficiency boost by only encode the video once
    generation_inputs = get_model_inputs(
        [c[:-1] for c in conversations], add_generation_prompt=True
    )

    return {
        **inputs,
        "labels": labels,
        "generation_inputs": generation_inputs,
    }


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metrics_dict = defaultdict(list)
    total_num_options = 0
    i = 0
    for pred, label in zip(preds, labels):
        label = set(c for c in label if c.isalpha())
        total_num_options += len(label)
        pred_options = set(c for c in pred if c.isalpha() and c.isupper())
        if len(pred_options) != len(pred):
            # invalid -> 0
            metrics_dict["per_question_accuracy"].append(0)
            metrics_dict["per_option_accuracy"].append(0)
            continue
        pred = pred_options

        per_question_accuracy = int(pred == label)
        # note: this doesn't penalize model outputting all options available
        per_option_accuracy = len(pred & label)
        metrics_dict["per_question_accuracy"].append(per_question_accuracy)
        metrics_dict["per_option_accuracy"].append(per_option_accuracy)

        if i % 100 == 0:
            print("-" * 100)
            print(f"Pred: {pred}")
            print(f"Label: {label}")
            print(f"PQA={per_question_accuracy:.4f} | POA={per_option_accuracy:.4f} ")
        i += 1

    return {
        "per_question_accuracy": float(np.mean(metrics_dict["per_question_accuracy"])),
        "per_option_accuracy": float(
            np.sum(metrics_dict["per_option_accuracy"]) / total_num_options
        ),
    }


eval_generation_config = GenerationConfig(
    max_length=8,
    do_sample=False,
)

# Configure training arguments using SFTConfig
training_args = Seq2SeqTrainingArguments(
    # data loading
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    # training schedule / optimization
    num_train_epochs=30,
    # max_steps=30,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    warmup_steps=5,
    learning_rate=2e-4,
    max_grad_norm=1.0,
    weight_decay=0.01,
    # eval
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_config=eval_generation_config,
    eval_strategy="steps",
    eval_steps=32,
    eval_on_start=True,
    # Logging / reporting
    output_dir="outputs_sft",
    logging_steps=1,
    report_to="wandb",
    # model saving
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
)


trainer = CausalPoolTrainer(
    model=model,
    args=training_args,
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_data_collator,
    compute_metrics=compute_metrics,
)

trainer_stats = trainer.train()
