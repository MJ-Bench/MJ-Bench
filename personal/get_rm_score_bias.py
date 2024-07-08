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

    data_list = []
    threshold = args.threshold

    data_path = './bias'
    # with open(os.path.join(data_path, 'eval_images/bias_dataset.json'), 'r', encoding='utf-8')as f:
    #     data = json.load(f)
    with open(os.path.join(data_path, f'eval_images/bias_dataset_{args.model}.json'), 'r', encoding='utf-8')as f:
        data = json.load(f)


    prompt_template = load_prompt_template(f"{args.prompt_template_path}")


    for id, example in tqdm(enumerate(data), desc="Evaluating RM", total=len(data)):

        if id < 618:
            continue
        
        image_path = os.path.join(data_path, f"{example['images_dir']}")
        caption = example["prompt"]

        prompt = prompt_template.format(caption=caption)

        rating, analysis = reward_model.get_score([image_path], prompt)


        print(f"Image: {image_path}, Rating: {rating}")
        print(f"Analysis: {analysis}")
        input()

        key = f"""{args.model}"""
        # example[key] = scores[0]
        example[key] = rating
        # example["gpt4score"] = rating
        example["analysis"] = analysis

        with open(os.path.join(data_path, f'eval_images/bias_dataset_{args.model}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    # save_dir = args.save_dir + f"{args.model}/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(f"{save_dir}/{args.dataset}{args.threshold}.json", 'w', encoding='utf-8') as f:
    #     json.dump(data_list, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="clipscore_v2", help="rm model to evaluate")
    parser.add_argument("--config_path", "-c", type=str, default="./config/config.yaml", help="config path")
    parser.add_argument("--dataset", type=str, default="nsfw", help="dataset")
    parser.add_argument("--local_buffer", type=str, default="cache/", help="local directory to buffer dataset")
    parser.add_argument("--save_dir", type=str, default="result/", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:1", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    parser.add_argument("--prompt_template_path", '-p', type=str,
                        default="single_image_number_scale5.txt", help="prompt template path")
    args = parser.parse_args()

    main(args)
