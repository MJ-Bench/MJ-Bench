from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from datasets import load_dataset
import json
import requests
from tqdm import tqdm
from io import BytesIO


def open_image(image, image_path=None):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    #
    return image



# if main
if __name__ == "__main__":
    dataset = load_dataset("yuvalkirstain/pickapic_v1", streaming=True)
    # if you want to download the latest version of pickapic download:
    # dataset = load_dataset("yuvalkirstain/pickapic_v2", num_proc=64)
    dataset = dataset['validation_unique']

    print("dataset", dataset)

    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    data_list = []
    threshold = 0.2

    for id, example in tqdm(enumerate(dataset)):
        new_item = {}

        image_0 = open_image(example["jpg_0"], example["image_0_url"])
        # save to dir = "/home/czr/DMs-RLAIF/dataset_buffer"
        image_0.save(f"/home/czr/DMs-RLAIF/dataset_buffer/{example['image_0_uid']}.jpg")
        image_1 = open_image(example["jpg_1"], example["image_1_url"])
        # save to dir = "/home/czr/DMs-RLAIF/dataset_buffer"
        image_1.save(f"/home/czr/DMs-RLAIF/dataset_buffer/{example['image_1_uid']}.jpg")
        data_list.append(new_item)

        # if id == 5:
        #     break

    # save_dir = "validation_pickscore_eval.json"

    # with open(save_dir, 'w') as f:
    #     json.dump(data_list, f, indent=4)
        