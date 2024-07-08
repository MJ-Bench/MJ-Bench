# import ImageReward as RM
# model = RM.load("ImageReward-v1.0")
#
# caption = 'an orange cat and a grey cat are lying together.'
# imgs_path = ["./clipscore/example/images/image1.jpg", "./clipscore/example/images/image2.jpg"]
# rewards = model.score(caption, imgs_path)
#
# print(rewards)
# --------------------------------------------------------------------------------
import ImageReward as RM
from PIL import Image
import torch
import json
from io import BytesIO
import clip
import warnings
from packaging import version
import sklearn.preprocessing
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import os
from transformers import BlipProcessor, BlipModel, AutoModel, AutoProcessor, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForImageTextRetrieval

def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image

def get_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred

def get_label(example):
    if example["label_0"] == 0.5:
        label = "tie"
    elif example["label_0"] == 1:
        label = "0"
    else:
        label = "1"
    return label

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = load_dataset("yuvalkirstain/pickapic_v1", streaming=True)
    dataset = dataset['validation_unique']

    image_buffer = "/home/czr/DMs-RLAIF/dataset/pickapic_v1/validation_unique"
    all_images = os.listdir(image_buffer)
    image_dict = {}
    for image_dir in all_images:
        image_id = image_dir.split(".jpg")[0]
        image_dict[image_id] = image_dir

    print("len of image_dict", len(image_dict))
    print("dataset", dataset)

    # Define save directory
    save_dir = "../result/validation_ImageReward_eval_0.00.json"
    data_list = []
    threshold = 0.0

    # load model
    model = RM.load("ImageReward-v1.0").to(device)

    for id, example in tqdm(enumerate(dataset)):
        # if id > 3:
        #     break

        new_item = {}

        # Calculate ImageReward
        image_0_path = os.path.join(image_buffer, image_dict[example["image_0_uid"]])
        image_1_path = os.path.join(image_buffer, image_dict[example["image_1_uid"]])
        imgs_path = [image_0_path, image_1_path]
        scores = model.score(example["caption"], imgs_path)
        # scores = np.array(scores).tolist()

        print(f"Image 0: {scores[0]}, Image 1: {scores[1]}")

        pred = get_pred(scores[0], scores[1], threshold)
        label = get_label(example)

        # Create new item
        new_item["id"] = id
        new_item["caption"] = example["caption"]
        new_item["ranking_id"] = example["ranking_id"]
        new_item["image_0_uid"] = example["image_0_uid"]
        new_item["image_1_uid"] = example["image_1_uid"]
        new_item["score_0"] = scores[0]
        new_item["score_1"] = scores[1]
        new_item["label"] = label
        new_item["pred"] = pred

        data_list.append(new_item)

    # Save data to JSON file
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
