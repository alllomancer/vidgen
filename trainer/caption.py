"""Caption every training item with Qwen2.5-VL: one .txt next to each clip/photo, starting
with the trigger word. Runs in the trainer venv as its own process (frees the GPU after).

    python caption.py <items dir> <trigger> <kind> <vlm repo>

What a caption leaves out matters as much as what it says: the concept being learned is
bound to the trigger word only if the caption does NOT describe it. So for a character the
caption skips the person's fixed looks, for a style it skips the look, for a motion it
names the setting and subject but not how the motion is performed.
"""

import os
import sys

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LEAVE_OUT = {
    "character": "Do not describe the person's face, hair, skin, body or identity; refer to them only as 'the person'.",
    "style": "Do not describe the visual style, color grading, film look or rendering; describe only the content.",
    "motion": "Name who and where, but do not describe how the movement is performed.",
    "place": "Do not describe the place's appearance in detail; name only what happens there.",
}


def main():
    items, trigger, kind, repo = sys.argv[1:5]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(repo)
    instruction = ("Write one caption (1-2 sentences, under 60 words) for training a video model: the setting, "
                   "the action, camera framing and movement, and lighting. " + LEAVE_OUT[kind] +
                   " Reply with the caption only.")
    names = sorted(n for n in os.listdir(items) if os.path.splitext(n)[1].lower() in VIDEO_EXTS | IMAGE_EXTS)
    for i, name in enumerate(names, 1):
        path = os.path.join(items, name)
        out = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(out):  # resumed job: already captioned
            continue
        if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
            media = {"type": "video", "video": f"file://{path}", "max_pixels": 360 * 640, "fps": 2.0}
        else:
            media = {"type": "image", "image": f"file://{path}", "max_pixels": 1024 * 1024}
        messages = [{"role": "user", "content": [media, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt",
                           **video_kwargs).to("cuda")
        ids = model.generate(**inputs, max_new_tokens=120)
        caption = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        caption = " ".join(caption.split()).strip().strip('"')
        with open(out, "w") as f:
            f.write(f"{trigger}, {caption}")
        print(f"CAPTION {i}/{len(names)} {name}: {caption[:100]}", flush=True)


if __name__ == "__main__":
    main()
