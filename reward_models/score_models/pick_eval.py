from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from datasets import load_dataset
import json
import requests
from tqdm import tqdm
from io import BytesIO


def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image

def calc_probs_for_dataset(ds, clip_model, clip_processor, device):
    probs = []
    for example in tqdm(ds):
        prob_0, prob_1 = infer_example(
            [example["jpg_0"], example["jpg_1"]],
            example["caption"],
            clip_model,
            clip_processor,
            device
        )
        probs.append((prob_0, prob_1))
    return probs


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


def calc_score(label, pred):
    if label == pred:
        score = 1
    elif "tie" in [label, pred]:
        score = 0.5
    else:
        score = 0
    return score


@torch.no_grad()
def infer_example(images, prompt, clip_model, clip_processor, device):
    images = [open_image(image) for image in images]

    image_inputs = clip_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = clip_processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_embs = clip_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = clip_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = clip_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()


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


    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


    data_list = []
    threshold = 0.0

    for id, example in tqdm(enumerate(dataset)):
        new_item = {}
        print(f"{id}: {example.keys()}")

        prob_0, prob_1 = infer_example(
                [example["jpg_0"], example["jpg_1"]],
                example["caption"],
                model,
                processor,
                device
            )
        print(f"prob_0: {prob_0}, prob_1: {prob_1}")
        label = get_label(example)
        pred = get_pred(prob_0, prob_1, threshold)

        new_item["id"] = id
        new_item["caption"] = example["caption"]
        new_item["ranking_id"] = example["ranking_id"]
        new_item["image_0_url"] = example["image_0_url"]
        new_item["image_1_url"] = example["image_1_url"]
        new_item["prob_0"] = prob_0
        new_item["prob_1"] = prob_1
        new_item["label"] = label
        new_item["pred_pick"] = pred

        data_list.append(new_item)

        # if id == 5:
        #     break

    save_dir = "../result/validation_pickscore_eval_0.00.json"

    with open(save_dir, 'w') as f:
        json.dump(data_list, f, indent=4)
        