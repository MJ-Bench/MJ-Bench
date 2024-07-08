import yaml
import sys, os
sys.path.append("./")
import argparse
# from reward_models import score_reward_models
# from reward_models import vlm_reward_models
from reward_models import closesource_models
from datasets import load_dataset
import json
from tqdm import tqdm
from utils.rm_utils import get_pred, get_label, get_config, open_image
import datetime

def load_prompt_template(path):
    with open(path, 'r') as f:
        prompt_template = f.read()
    return prompt_template

def load_json_dataset(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset


def main(args):
    reward_models_config = get_config(args.config_path, "reward_models")

    rm_type_dict = {}
    for parent_label, sub_models in reward_models_config.items():
        for sub_model in sub_models:
            rm_type_dict[sub_model] = parent_label
    
    device = args.device

    print(args.model)


    # dataset = load_dataset(args.dataset, streaming=True)
    # if you want to download the latest version of pickapic download:
    # dataset = load_dataset("yuvalkirstain/pickapic_v2", num_proc=64)
    # dataset = dataset['validation_unique']

    if os.path.exists(args.local_buffer):
        root_buffer = args.local_buffer
        all_images = os.listdir(root_buffer)
        image_dict = {image_dir.split(".jpg")[0]: image_dir for image_dir in all_images}
    else:
        image_dict = {}

    if rm_type_dict[args.model] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = score_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.model] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = vlm_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.model] == "closesource_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = closesource_models.Scorer(args.model, model_config["model_name"], model_config["api_key"], model_config["base_url"])
    else:
        raise ValueError(f"Model {args.model} not found in config file")


    # with open("/home/czr/MM-Reward/online_result/gpt-4o_alignment_number10.json", 'r') as f:
    #     data_list = json.load(f)


    data_list = []
    threshold = args.threshold

    try:
        dataset = load_dataset(args.dataset, streaming=True)
        dataset = dataset['test_unique']
    except:
        dataset = load_json_dataset(args.dataset)


    prompt_template = load_prompt_template(f"{args.prompt_template_path}")

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id, example in tqdm(enumerate(dataset), desc="Evaluating RM", total=len(dataset)):

        # if id < 13:
        #     continue
        new_item = {}

        # print(image_dict)

        # if example["image0"] in image_dict and example["image1"] in image_dict:
        image_0_path = os.path.join(root_buffer, example["image0"])
        image_1_path = os.path.join(root_buffer, example["image1"])

        caption = example["caption"]
        label = example['label']
        

        prompt = prompt_template.format(caption=caption)


        image_1_rating, image_2_rating, preference, analysis = reward_model.get_score([image_0_path, image_1_path], prompt)


        print(f"Image 0: {image_0_path} , Image 1: {image_1_path}, Rating 0: {image_1_rating}， Rating 1: {image_2_rating}")
        print(f"Analysis: {analysis}")
        # input()

        new_item["id"] = example["id"]
        new_item["caption"] = caption
        new_item["image_0_uid"] = example["image0"]
        new_item["image_1_uid"] = example["image1"]
        new_item["score_0"] = image_1_rating
        new_item["score_1"] = image_2_rating
        new_item["label"] = label
        new_item["vlm_pred"] = preference
        new_item["output"] = analysis

        data_list.append(new_item)

        with open(save_dir+f"/{args.model}_alignment_number10.json", 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="clipscore_v2", help="rm model to evaluate")
    parser.add_argument("--config_path", "-c", type=str, default="./config/config.yaml", help="config path")
    parser.add_argument("--dataset", type=str, default="nsfw", help="dataset")
    parser.add_argument("--local_buffer", type=str, default="/home/czr/MM-Reward/alignment/benchmark/images", help="local directory to buffer dataset")
    parser.add_argument("--save_dir", type=str, default="online_result/", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:1", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    parser.add_argument("--prompt_template_path", '-p', type=str,
                        default="/home/czr/MM-Reward/prompt_template/prompts_multi_image/alignment_multi_number_scale10.txt", help="prompt template path")
    args = parser.parse_args()

    main(args)
