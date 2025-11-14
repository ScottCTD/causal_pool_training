from transformers import AutoProcessor

model_name = "Qwen/Qwen3-VL-4B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)

messages = [[
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "test_video.mp4",
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 256 * 32 * 32,
                "total_pixels": 20480 * 32 * 32,
            },
            {"type": "text", "text": "Describe ball movements in great details."},
        ],
    },
],
            
[
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "test_video_2.mp4",
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 256 * 32 * 32,
                "total_pixels": 20480 * 32 * 32,
            },
            {"type": "text", "text": "Describe ball movements in great details."},
        ],
    },
],
            ]

from qwen_vl_utils import process_vision_info

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
images, videos, video_kwargs = process_vision_info(
    messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
)

# split the videos and according metadatas
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

# since qwen-vl-utils has resize the images/videos, \
# we should pass do_resize=False to avoid duplicate operation in processor!
inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, padding=True, padding_side="left", **video_kwargs)

print(processor.tokenizer.pad_token_id)
print(inputs["input_ids"].shape)

print(inputs["input_ids"][0][-10:])
print(inputs["input_ids"][1][-10:])