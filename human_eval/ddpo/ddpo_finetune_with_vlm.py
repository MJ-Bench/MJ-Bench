# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python experimental/ddpo_finetune.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=16 \
    --train_batch_size=8 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
    --use_lora=True
"""
import os
from dataclasses import dataclass, field

from PIL import Image
from io import BytesIO
import yaml
import collections
import re
import logging

import wandb
import datetime
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser


#from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
# from trl.import_utils import is_npu_available, is_xpu_available

import sys
sys.path.append('/cpfs01/user/duyichao/workspace/LLM_RLAIF/MRM-Bench')
sys.path.append('/cpfs01/user/duyichao/workspace/LLM_RLAIF/MRM-Bench/trl_modified/trl')
sys.path.append("/cpfs01/user/duyichao/workspace/LLM_RLAIF/MRM-Bench/trl_modified/trl/trainer")
sys.path.append("/cpfs01/user/duyichao/workspace/LLM_RLAIF/MRM-Bench/trl_modified/trl/models")
from trainer import DDPOConfig
from ddpo_trainer import DDPOTrainer
from modeling_sd_base import DefaultDDPOStableDiffusionPipeline
from import_utils import is_npu_available, is_xpu_available

# sys.path.append("utils")
# from rm_utils import get_pred, get_label, get_config, open_image

from reward_models import score_reward_models
from reward_models import vlm_reward_models

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ddpo_finetune_with_vlm")


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


def open_image(image, image_path=None):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    #
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

def get_config(config_path, key=None):

    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        if key:
            return config_data[key]
        else:
            return config_data

def load_rm_prompt_template(path):
    with open(path, 'r') as f:
        prompt_template = f.read()
    return prompt_template


eng2score = collections.OrderedDict({
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10"
})

narrative2score = {
    5: collections.OrderedDict(
        {"extremely poor": 1, "poor": 2, "average": 3, "good": 4, "outstanding": 5}
    ),
    10: collections.OrderedDict(
        {"extremely poor": 1, "very poor": 2, "poor": 3, "above average": 4, "above average": 6, "average": "5", "very good": 8, "good": 7, "excellent": 9, "outstanding": 10}
    ),
}






def parse_rm_output(args, text, pattern, split_text="RATING", aid_text="assistant"):
    text = text.split(aid_text)[-1]
    result_match = re.findall(pattern, text)
    if  len(result_match) != 0:
        if args.metric_type == "narrative":
            return narrative2score[args.metric_scale][str(result_match[-1]).lower()]
        else:
            return float(result_match[-1])
        
    if args.metric_type == "narrative":
        text = text.split(split_text)[-1].lower()  
        for key in narrative2score[args.metric_scale]: # TODO key可能是某个单词中间的几个字母
            if key in text:
                score = narrative2score[args.metric_scale][key]
                return float(score)
   
    match = re.search(r"(\d?\.\d+|\d+)", text)
    if match:
        return float(match.group())
        
    words = text.split()
    for key in eng2score:
        if key in words:
            return float(eng2score[key])
    return None
    

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use."})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    reward_model_device: str = field(default="cuda", metadata={"help": "reward_model_device of reward model"})
    reward_model_name: str = None
    prompt_file: str = None
    reward_model_prompt_file: str = None
    config_path: str = field(default="config/config.yaml", metadata={"help": "the configuration for reward model."})
    save_dir: str = field(default="result/", metadata={"help": "the directory to save the result."})
    perspective: str = field(default="alignment", metadata={"help": "the perspective."})
    metric_scale: int = field(default=10, metadata={"help": "the scale of the metric."})
    metric_type: str = field(default="number", metadata={"help": "the type of the metric."})

def get_reward_fn(args, **kwargs):

    reward_models_config = get_config(args.config_path, "reward_models")

    rm_type_dict = {}
    for parent_label, sub_models in reward_models_config.items():
        for sub_model in sub_models:
            rm_type_dict[sub_model] = parent_label
    
    reward_model_device = args.reward_model_device

    if rm_type_dict[args.reward_model_name] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.reward_model_name]][args.reward_model_name]
        reward_model = score_reward_models.Scorer(args.reward_model_name, model_config["model_path"], model_config["processor_path"], reward_model_device)
    elif rm_type_dict[args.reward_model_name] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.reward_model_name]][args.reward_model_name]
        reward_model = vlm_reward_models.Scorer(args.reward_model_name, model_config["model_path"], model_config["processor_path"], reward_model_device)
        prompt_template = load_rm_prompt_template(args.reward_model_prompt_file)
    else:
        raise ValueError(f"Model {args.model} not found in config file")
    
    
    metric_pattern0 ="|".join(narrative2score[args.metric_scale])
    metric_pattern1 = "|".join([f'\[{metric}\]' for metric in narrative2score[args.metric_scale]])
    metric_pattern2 = "|".join([f'\[{metric.lower()}\]' for metric in narrative2score[args.metric_scale]])
    metric_pattern3 = "|".join([f'\[{metric.upper()}\]' for metric in narrative2score[args.metric_scale]])
    metric_pattern4 = "|".join([f'\"{metric}\"' for metric in narrative2score[args.metric_scale]])
    metric_pattern5 = "|".join([f'\"{metric.lower()}\"' for metric in narrative2score[args.metric_scale]])
    metric_pattern6 = "|".join([f'\"{metric.upper()}\"' for metric in narrative2score[args.metric_scale]])
    metric_pattern = re.compile(r'RATING: (\d+\.+\d+|\d+|{0}|{1}|{2}|{3}{4}{5}{6})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3, metric_pattern4, metric_pattern5, metric_pattern6))
    
    def wrapper_reward_fn(images, prompts, metadata):
        reward_list = []
        for image, prompt in zip(images, prompts):
            image = image.to(reward_model_device)
            if rm_type_dict[args.reward_model_name] == "score_models":
                score = reward_model.get_score([image], prompt)
                
            elif rm_type_dict[args.reward_model_name] == "opensource_vlm":
                prompt = prompt_template.format(caption=prompt) # TODO caption 
                score = reward_model.get_score([image], prompt)
                score = [parse_rm_output(args, score[0], pattern=metric_pattern, aid_text=model2aid[args.reward_model_name]) / args.metric_scale]
                # print("reward score is:", score)
            else:
                raise ValueError(f"Model {args.model} not found in config file")
            reward_list.append(score)
        return reward_list, None

    return wrapper_reward_fn
    

def prompt_fn(prompts):
    return np.random.choice(prompts), None

def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    logger.info(f"ARGS: {args}")
    logger.info(f"DDPO_CONFIG: {ddpo_config}")
    
    wandb.login(key="")
    for i in range(10):
        try:
            wandb.init(entity='my-team', project=f"{args.tracker_project_name}", name=f"{args.perspective}_{args.reward_model_name}_{datetime.datetime.now()}", reinit=True)
        except:
            logger.info('wandb init failed')
            continue

    
    ddpo_config.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "project_dir": args.save_dir,
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=args.use_lora
    )


    with open(args.prompt_file, "r") as f:
        prompts = f.readlines()
    
        
    from functools import partial
    prompt_fn = partial(prompt_fn, prompts=prompts)

    trainer = DDPOTrainer(
        ddpo_config,
        get_reward_fn(args),
        prompt_fn,
        pipeline,
        #image_samples_hook=image_outputs_logger,
    )

    trainer.train()
