import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint with base model")
    parser.add_argument("-b", "--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("-o", "--out_dir", type=str, default=None, help="Output directory for merged model (default: {ckpt_path}/merged)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to load models on: 'auto' (use GPU if available), 'cpu', or 'cuda' (default: auto)")

    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = f"{args.ckpt_path}/merged"

    # Determine device_map based on device argument
    if args.device == "cpu":
        device_map = "cpu"
    elif args.device == "cuda":
        device_map = "cuda"
    else:  # auto
        import torch
        device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    print(f"Loading base model from {args.base_model} on {device_map}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, 
        dtype="bfloat16",
        device_map=device_map,
    )

    print(f"Loading LoRA adapter from {args.ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, args.ckpt_path)

    print("Merging and unloading...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.out_dir}...")
    merged.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    print("Done!")

if __name__ == "__main__":
    main()
