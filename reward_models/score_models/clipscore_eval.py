from transformers import AutoProcessor, AutoModel
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
from transformers import AutoProcessor, AutoModel

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

def get_clipscore(images, caption, model, transform, device, w=2.5):
    # images = Image.open(images)
    images = open_image(images)
    images = transform(images).unsqueeze(0).to(device)

    with torch.no_grad():
        text_inputs = clip.tokenize([caption[:77]]).to(device)
        caption_emb = model.encode_text(text_inputs).cpu().numpy()
        image_emb = model.encode_image(images).cpu().numpy()

        if version.parse(np.__version__) < version.parse('1.21'):
            image_emb = sklearn.preprocessing.normalize(image_emb, axis=1)
            caption_emb = sklearn.preprocessing.normalize(caption_emb, axis=1)
        else:
            warnings.warn(
                'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
            image_emb = image_emb / np.sqrt(np.sum(image_emb**2, axis=1, keepdims=True))
            caption_emb = caption_emb / np.sqrt(np.sum(caption_emb**2, axis=1, keepdims=True))

        per = w * np.clip(np.sum(image_emb * caption_emb, axis=1), 0, None).astype(np.float32)
        return per

if __name__ == "__main__":
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # Load dataset
    dataset = load_dataset("yuvalkirstain/pickapic_v1", streaming=True)
    dataset = dataset['validation_unique']

    image_buffer = "/home/czr/DMs-RLAIF/dataset/pickapic_v1/validation_unique"
    # get all the images in the buffer
    all_images = os.listdir(image_buffer)
    image_dict = {}
    for image_dir in all_images:
        image_id = image_dir.split(".jpg")[0]
        image_dict[image_id] = image_dir

    print("len of image_dict", len(image_dict))
    print("dataset", dataset)

    # Define save directory
    save_dir = "../result/validation_clipscore_eval_0.00.json"
    data_list = []
    threshold = 0.0

    for id, example in tqdm(enumerate(dataset)):

        # if id > 3:
        #     break

        new_item = {}
        # print(f"{id}: {example.keys()}")
        print(example["caption"])
        print(example["caption"][:77])

        # Calculate CLIPScore for image 0
        image_0_path = os.path.join(image_buffer, image_dict[example["image_0_uid"]])
        image_0 = Image.open(image_0_path)
        score_0 = get_clipscore(image_0, example["caption"], model, transform, device).tolist()

        # Calculate CLIPScore for image 1
        image_1_path = os.path.join(image_buffer, image_dict[example["image_1_uid"]])
        image_1 = Image.open(image_1_path)
        score_1 = get_clipscore(image_1, example["caption"], model, transform, device).tolist()

        pred = get_pred(score_0[0], score_1[0], threshold)
        label = get_label(example)

        # Create new item
        new_item["id"] = id
        new_item["caption"] = example["caption"]
        new_item["ranking_id"] = example["ranking_id"]
        new_item["image_0_uid"] = example["image_0_uid"]
        new_item["image_1_uid"] = example["image_1_uid"]
        new_item["score_0"] = score_0[0]  # Convert numpy array to list
        new_item["score_1"] = score_1[0]  # Convert numpy array to list
        new_item["label"] = label
        new_item["pred"] = pred

        data_list.append(new_item)

    # Save data to JSON file
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
