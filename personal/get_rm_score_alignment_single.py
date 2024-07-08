import sys  # NOQA: E402
sys.path.append("./")  # NOQA: E402

import datetime
import logging
from tqdm import tqdm
import json
from datasets import load_dataset
from reward_models import score_reward_models, vlm_reward_models
import argparse
import yaml
import os
import re


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("alignment_rm_score")


def load_prompt_template(path):
    with open(path, 'r') as f:
        prompt_template = f.read()
    return prompt_template


def load_json_dataset(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset


def open_image(image, image_path=None):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image
    

def get_score_model_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred

def get_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred


def get_config(config_path, key=None):
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        if key:
            return config_data[key]
        else:
            return config_data


def parse_vlm_output(text, patterns, aid_text="assistant"):
    text = text.split(aid_text)[-1]
    results = dict()
    for k in patterns.keys():
        pattern = patterns[k]
        result_match = re.findall(pattern, text)
        if len(result_match) != 0:
            results[k] = result_match[-1]
        else:
            results[k] = "N/A"
    return text, results

model2aid = {
    "llava-1.5-7b-hf": "ASSISTANT",
    "llava-1.5-13b-hf": "ASSISTANT",
    "llava-v1.6-mistral-7b-hf": "[/INST]",
    "llava-v1.6-vicuna-13b-hf": "ASSISTANT",
    "llava-v1.6-34b-hf": "assistant",
    "internvl-chat-v1-2-plus": "assistant",
    "internvl-chat-v1-5": "internvl-chat-v1-5",
    "instructblip": "instructblip",
    "minigpt4": "minigpt4",
    "idefics2-8b": "Assistant",
    "qwen-vl-chat": "qwen-vl-chat"
}

metric_narratives = {
    5: ["Extremely Poor", "Poor", "Average", "Good", "Outstanding"],
    7: ["Extremely Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Outstanding"],
    10: ["Extremely Poor", "Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good", "Excellent", "Outstanding"]
}


def main(args):
    metric_type = args.metric_type
    metric_scale = args.metric_scale
    perspective = args.perspective
    multi_image = args.multi_image
    

    narrative2number = {metric: i + 1 for i, metric in enumerate(metric_narratives[metric_scale])}
    
    metric_pattern0 ="|".join(metric_narratives[metric_scale])
    metric_pattern1 = "|".join([f"\[{metric}\]" for metric in metric_narratives[metric_scale]])
    metric_pattern2 = "|".join([f"\[{metric.lower()}\]" for metric in metric_narratives[metric_scale]])
    metric_pattern3 = "|".join([f"\[{metric.upper()}\]" for metric in metric_narratives[metric_scale]])
    if multi_image:
        parse_patterns = {
            "IMAGE-1 RATING": re.compile(r'IMAGE-1 RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3)),
            "IMAGE-2 RATING": re.compile(r'IMAGE-2 RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3)),
            "BETTER IMAGE": re.compile(r'BETTER IMAGE: ([0-2]|IMAGE-1|IMAGE-2|IMAGE 1|IMAGE 2|image-1|image-2|image 1|image 2)'),
        }
    else:
        parse_patterns = {
            "RATING": re.compile(r'{4}-RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3, perspective.replace("_", "-").upper())),
            "ANASYLIS": re.compile(r'ANASYLIS: (.+)'),
        }
    
    
    reward_models_config = get_config(args.config_path, "reward_models")

    rm_type_dict = {}
    for parent_label, sub_models in reward_models_config.items():
        for sub_model in sub_models:
            rm_type_dict[sub_model] = parent_label

    device = args.device
    try:
        dataset = load_dataset(args.dataset, streaming=True)
        dataset = dataset['test_unique']
    except:
        dataset = load_json_dataset(args.dataset)

    if os.path.exists(args.local_buffer):
        root_buffer = args.local_buffer
        all_images = os.listdir(root_buffer)
        image_dict = {image_dir: image_dir for image_dir in all_images}
    else:
        image_dict = {}

    logger.info(f"Loading reward model form {args.model}...")
    if rm_type_dict[args.model] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = score_reward_models.Scorer(
            args.model, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.model] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = vlm_reward_models.Scorer(
            args.model, model_config["model_path"], model_config["processor_path"], device, multi_image=multi_image)
    else:
        raise ValueError(f"Model {args.model} not found in config file")
    
    data_list = []
    threshold = args.threshold

    logger.info(f"Loading prompt template from {args.prompt_template_path}")
    prompt_template = load_prompt_template(f"{args.prompt_template_path}")

    for id, example in tqdm(enumerate(dataset), desc="Evaluating RM: "):
        # if id > 1: break
        new_item = {}

        if example["image0"] in image_dict and example["image1"] in image_dict:
            image_0_path = os.path.join(root_buffer, example["image0"])
            image_1_path = os.path.join(root_buffer, example["image1"])

        caption = example["caption"]
        label = example['label']
        

        prompt = prompt_template.format(caption=caption)

        if rm_type_dict[args.model] == "score_models":
            scores = reward_model.get_score([image_0_path, image_1_path], caption)
            score_0, score_0 = scores[0], scores[1]
            output_0, output_1 = scores[0], scores[1]
        elif "opensource_vlm" in rm_type_dict[args.model]:
            scores = reward_model.get_score([image_0_path, image_1_path], prompt)
            if multi_image:
                vlm_output, parsed_results = parse_vlm_output(scores, parse_patterns, aid_text=model2aid[args.model])
                score_0 = parsed_results["IMAGE-1 RATING"]
                score_1 = parsed_results["IMAGE-2 RATING"]
                better_id = parsed_results["BETTER IMAGE"]
                if better_id.upper() == "IMAGE-1" or better_id.upper() == "IMAGE 1":
                    better_id = 1
                elif better_id.upper() == "IMAGE-2" or better_id.upper() == "IMAGE 2":
                    better_id = 2
            else:
                output_0, parsed_results_0 = parse_vlm_output(scores[0], parse_patterns, aid_text=model2aid[args.model])
                score_0 = parsed_results_0["RATING"]
                output_1, parsed_results_1 = parse_vlm_output(scores[1], parse_patterns, aid_text=model2aid[args.model])
                
                score_1 = parsed_results_1["RATING"]   
        else:
            raise ValueError(f"Model {args.model} not found in config file")
        try:
            pred = "N/A"
            if metric_type == "narrative":
                score_0 = score_0.replace("[", "").replace("]", "").title()
                score_1 = score_1.replace("[", "").replace("]", "").title()
                score_0 = narrative2number[score_0]
                score_1 = narrative2number[score_1]
            pred = get_pred(float(score_0), float(score_1), threshold) 
        except:
            logger.info(f"Cannot extract score from vlm output. sample id is {id}, score_0 is {score_0}, score_1 is {score_1}")
     
        if multi_image:
            try:
                better_id = int(better_id) - 1
            except:
                logger.info(f"Cannot extract BETTER_IMAGE. sample id is {id}, better_id is {better_id}")
                
            new_item["id"] = example["id"]
            new_item["caption"] = caption
            new_item["image_0_uid"] = example["image0"]
            new_item["image_1_uid"] = example["image1"]
            new_item["score_0"] = score_0
            new_item["score_1"] = score_1
            new_item["label"] = label
            new_item["vlm_pred"] = better_id
            new_item["score_pred"] = pred
            new_item["output"] = vlm_output
        else:
            new_item["id"] = example["id"]
            new_item["caption"] = caption
            new_item["image_0_uid"] = example["image0"]
            new_item["image_1_uid"] = example["image1"]
            new_item["score_0"] = score_0
            new_item["score_1"] = score_1
            new_item["label"] = label
            new_item["pred"] = pred
            new_item["output_0"] = output_0
            new_item["output_1"] = output_1

        data_list.append(new_item)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info(f"Saving result to {save_dir}")
    with open(save_dir+f"/{args.model}_{metric_type}_scale{metric_scale}.json", 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str,
                        default="clipscore_v1", help="rm model to evaluate")
    parser.add_argument("--config_path", "-c", type=str,
                        default="config/config.yaml", help="config path")
    parser.add_argument("--prompt_template_path", type=str,
                        default="single_image_number_scale5.txt", help="prompt template path")
    parser.add_argument("--dataset", type=str,
                        default="yuvalkirstain/pickapic_v1", help="dataset")
    parser.add_argument("--local_buffer", type=str, default="cache/",
                        help="local directory to buffer dataset")
    parser.add_argument("--save_dir", type=str,
                        default="result/", help="save directory")
    parser.add_argument("--device", type=str,
                        default="cuda:0", help="cuda or cpu")
    parser.add_argument("--threshold", type=float,
                        default=0.0, help="threshold")
    parser.add_argument("--metric_type", type=str, default="number",
                        choices=['number', 'narrative', 'logits'], help="metric scalar")
    parser.add_argument("--metric_scale", type=int, default=5,
                        choices=[1, 5, 7, 10], help="metric scalar")
    parser.add_argument("--perspective", default="alignment", choices=['alignment', 'bias & fairness', 'safety', 'artifact'], help="evaluation perspective")
    parser.add_argument("--multi_image", action="store_true", help="multi image input")
    args = parser.parse_args()

    main(args)
