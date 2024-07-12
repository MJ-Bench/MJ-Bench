import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import cv2 as cv
from tqdm import tqdm
import json


def dummy(images, **kwargs): 
    return images, [False] * len(images)

# Open the file in read mode
file_path = 'dpo_data/train_alignment.txt'

base_dir = "/home/czr/MM-Reward/dpo_data/alignment"

# Read the file line by line and print each line
with open(file_path, 'r') as file:
    prompt_list = file.readlines()

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.safety_checker = dummy
pipe = pipe.to("cuda")

alignment_image_pair = []

data_list = []

for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
    new_item = {}
    prompt = prompt.strip()
    print(prompt)

    new_item["prompt"] = prompt

    image = pipe(prompt).images[0] 
    image_1_dir = f"idx_{idx}_s1.png"
    image.save(base_dir + "/+" + image_1_dir)

    image = pipe(prompt).images[0] 
    image_2_dir = f"idx_{idx}_s2.png"
    image.save(base_dir + "/+" + image_2_dir)

    new_item["image_1"] = image_1_dir
    new_item["image_2"] = image_2_dir

    data_list.append(new_item)

    # Save the data_list
    with open('dpo_data/image_pair.json', 'w') as f:
        json.dump(data_list, f, indent=4)


