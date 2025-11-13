from collections import defaultdict
from typing import Dict
from datasets import load_dataset
import numpy as np

dataset_name = "trl-lib/llava-instruct-mix"
train_dataset = load_dataset(dataset_name, split="train[:10%]")
from qwen_vl_utils import process_vision_info


from transformers import EvalPrediction, GenerationConfig, Qwen3VLForConditionalGeneration, AutoProcessor, Seq2SeqTrainingArguments
import torch

from custom_trainer import CausalPoolTrainer

model_name = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer


from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        # Text tower
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
        # Vision tower (attention and MLP)
        'qkv', 'proj', 'linear_fc1', 'linear_fc2',
    ],
)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

import torch

def index_to_letter(index):
    assert 0 <= index < 26, f"Index must be between 0 and 25, got {index}"
    return chr(index + ord("A"))

def letter_to_index(letter):
    assert len(letter) == 1 and letter.isalpha(), f"Letter must be a single alphabetic character, got {letter}"
    return ord(letter.upper()) - ord("A")

def get_assistant_mask(input_ids):
    # Vectorized search for sequence "<|im_start|>assistant\n" -> [151644, 77091, 198]
    pattern = torch.tensor([151644, 77091, 198], device=input_ids.device, dtype=input_ids.dtype)
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
    mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) >= start_after.unsqueeze(1)
    return mask

def get_model_inputs(conversations):
    # TODO: verify paddings
    text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    images, videos, video_kwargs = process_vision_info(conversations, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    # split the videos and according metadatas
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # since qwen-vl-utils has resize the images/videos,
    # we should pass do_resize=False to avoid duplicate operation in processor!
    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

    return inputs

def video_assistant_data_collator(samples):
    conversations = []
    for sample in samples:
        question = sample["question"]
        choices = sample["choices"]
        
        question_prompt = f"{question}\n\n"
        for i, choice in enumerate(choices):
            question_prompt += f"{index_to_letter(i)}. {choice}\n"
        
        ground_truth_indices = sample["ground_truth"]  # List[int] indicies
        ground_truth_answer = "".join(index_to_letter(i) for i in ground_truth_indices)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sample["video"],
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
            }
        ]
        conversations.append(conversation)
    
    inputs = get_model_inputs(conversations)
    
    # construct labels
    input_ids = inputs.input_ids
    # TODO: there's a bug here, the assistant mask doesn't mask the turn end token
    assistant_mask = get_assistant_mask(input_ids)
    labels = inputs.input_ids.clone()
    labels[~assistant_mask] = -100
    
    return {
        **inputs,
        "labels": labels,
    }


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    assistant_mask = label_ids != -100
    label_ids = np.where(assistant_mask, label_ids, tokenizer.pad_token_id)
    pred_ids = pred_ids[assistant_mask]
    
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metrics_dict = defaultdict(list)
    i = 0
    for pred, label in zip(preds, labels):
        token_level_accuracy = np.mean(pred == label)
        metrics_dict["token_level_accuracy"].append(token_level_accuracy)
        
        if i % 32 == 0:
            print("-" * 100)
            print(f"Pred: {pred}")
            print(f"Label: {label}")
            print(
                f"TLA={token_level_accuracy:.4f} "
            )
        i += 1

    return {
        "mean_token_accuracy": float(np.mean(metrics_dict["token_level_accuracy"])),
    }

eval_generation_config = GenerationConfig(
    max_length=8,
    do_sample=False,
)

output_dir = "outputs/"

# Configure training arguments using SFTConfig
training_args = Seq2SeqTrainingArguments(
    # training schedule / optimization
    #num_train_epochs=1,
    max_steps=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    warmup_steps=5,
    learning_rate=2e-4,
    max_grad_norm=1.0,
    weight_decay=0.01,
    max_length=None,  # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it

    # eval
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_config=eval_generation_config,
    eval_strategy="steps",
    eval_steps=128,
    eval_on_start=True,

    # Logging / reporting
    output_dir=output_dir,
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
    train_dataset=train_dataset,
    # data_collator=video_assistant_data_collator,
    # compute_metrics=compute_metrics,
)

trainer_stats = trainer.train()