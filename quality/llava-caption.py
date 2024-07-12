import requests
from PIL import Image
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
from tqdm import tqdm

model_id = "llava-hf/llava-1.5-7b-hf"

prompt = "USER: <image>\nWrite a simple caption for this image.\nASSISTANT:"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

image_dir = "./blur_dataset/sharp"
all_images = os.listdir(image_dir)

json_path = "./blur_dataset/captions_blur.json"
with open(json_path, "r", encoding='utf-8')as f:
    captions_data = json.load(f)

for image_name in tqdm(all_images[100:], desc="Processing", leave=True):
    print(f"Processing image: {image_name}")
    image_path = os.path.join(image_dir, image_name)
    image_data = Image.open(image_path)
    inputs = processor(prompt, image_data, return_tensors='pt').to(0, torch.float16)#, torch.float16
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    output = processor.decode(output[0][2:], skip_special_tokens=True)
    caption = output.split("ASSISTANT:")[1].strip()

    print(f"Caption: {caption}\n")

    captions_data.append({
        "image_name": image_name,
        "caption": caption
    })

    # Save captions_data to a JSON file
    output_json_file = "./blur_dataset/captions_blur.json"
    with open(output_json_file, "w", encoding='utf-8') as json_file:
        json.dump(captions_data, json_file, indent=4, ensure_ascii=False)

print("Captions data have been saved")
