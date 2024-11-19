import yaml
import sys, os
sys.path.append("./")
import argparse
from reward_models import score_reward_models
from reward_models import vlm_reward_models
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

    if "multi" in args.prompt_template_path:
        assert args.multi_image == True, "Please use multi image prompt template"
    if "single" in args.prompt_template_path:
        assert args.multi_image == False, "Please use single image prompt template"
    
    if args.multi_image:
        kwargs = {"multi_image": True}
    else:
        kwargs = {"multi_image": False}

    if rm_type_dict[args.model] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = score_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.model] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = vlm_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device, **kwargs)
    elif rm_type_dict[args.model] == "closesource_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = closesource_models.Scorer(args.model, model_config["model_name"], model_config["api_key"], model_config["base_url"])
    else:
        raise ValueError(f"Model {args.model} not found in config file")


    data_list = []
    threshold = args.threshold

    try:
        dataset = load_dataset(args.dataset, streaming=True)
        dataset = dataset[args.perspective]
    except:
        dataset = load_json_dataset(args.dataset)
        dataset = dataset[args.perspective]


    prompt_template = load_prompt_template(f"{args.prompt_template_path}")
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id, example in tqdm(enumerate(dataset), desc="Evaluating RM"):

        new_item = {}

        image_0_path = "cache/image0.jpg"
        image_1_path = "cache/image1.jpg"
        example["image0"].save(image_0_path)
        example["image1"].save(image_1_path)

        caption = example["caption"]
        label = example['label']
        
        prompt = prompt_template.format(caption=caption)

        print(f"Caption: {caption}")

        if rm_type_dict[args.model] == "closesource_models" or args.multi_image:
            responses = reward_model.get_score([image_0_path, image_1_path], prompt)
            image_1_rating = responses.split("IMAGE-1 RATING:")[1].split("IMAGE-2 RATING:")[0]
            image_2_rating = responses.split("IMAGE-2 RATING:")[1].split("BETTER IMAGE:")[0]
            preference = responses.split("BETTER IMAGE:")[1].split("ANALYSIS OF CHOICE:")[0]
            analysis = responses.split("ANALYSIS OF CHOICE:")[1]

            print(f"Analysis: {analysis}")
        else:
            responses = reward_model.get_score([image_0_path, image_1_path], prompt)
            image_1_rating, image_2_rating = responses
            preference = get_pred(image_1_rating, image_2_rating, threshold)
            analysis = None
        
        print(f"Image 0: {image_0_path} , Image 1: {image_1_path}, Rating 0: {image_1_rating}, Rating 1: {image_2_rating}")

        new_item["caption"] = caption
        new_item["image_0_uid"] = str(example["image0"])
        new_item["image_1_uid"] = str(example["image1"])
        new_item["score_0"] = image_1_rating
        new_item["score_1"] = image_2_rating
        new_item["label"] = label
        new_item["vlm_pred"] = preference
        new_item["analysis"] = analysis
        

        data_list.append(new_item)

        with open(f"{save_dir}/{args.model}_{args.perspective}.json", 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="clipscore_v2", help="judge model to evaluate")
    parser.add_argument("--config_path", "-c", type=str, default="./config/config.yaml", help="config path")
    parser.add_argument("--dataset", type=str, default="MJ-Bench/MJ-Bench", help="dataset id")
    parser.add_argument("--perspective", type=str, default="alignment", help="subset to evaluate")
    parser.add_argument("--save_dir", type=str, default="result/", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    parser.add_argument("--multi_image", "-mu", action="store_true", default=False, help="single image or multi image")
    parser.add_argument("--prompt_template_path", '-p', type=str,
                        default="prompt_template/prompts_multi_image/alignment_multi_number_scale10.txt", help="prompt template path")
    args = parser.parse_args()

    main(args)
