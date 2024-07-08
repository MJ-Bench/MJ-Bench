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
import sys, os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser


#from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
# from trl.import_utils import is_npu_available, is_xpu_available

import sys
sys.path.append("./")
sys.path.append('trl_modified/trl')
sys.path.append("trl_modified/trl/trainer")
sys.path.append("trl_modified/trl/model")
from trainer import DDPOConfig
from ddpo_trainer import DDPOTrainer
from modeling_sd_base import DefaultDDPOStableDiffusionPipeline
from import_utils import is_npu_available, is_xpu_available

sys.path.append("utils")
from reward_models import score_reward_models
from reward_models import vlm_reward_models


from rm_utils import get_pred, get_label, get_config, open_image

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use."})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    device: str = field(default="cuda", metadata={"help": "The device to use."})
    reward_model_name: str = None
    prompt_file: str = None
    config_path: str = field(default="config/config.yaml", metadata={"help": "the configuration for reward model."})
    save_dir: str = field(default="result/", metadata={"help": "the directory to save the result."})

def get_reward_fn(args, **kwargs):

    reward_models_config = get_config(args.config_path, "reward_models")

    rm_type_dict = {}
    for parent_label, sub_models in reward_models_config.items():
        for sub_model in sub_models:
            rm_type_dict[sub_model] = parent_label
    
    device = args.device

    if rm_type_dict[args.reward_model_name] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.reward_model_name]][args.reward_model_name]
        reward_model = score_reward_models.Scorer(args.reward_model_name, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.reward_model_name] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.reward_model_name]][args.reward_model_name]
        reward_model = vlm_reward_models.Scorer(args.reward_model_name, model_config["model_path"], model_config["processor_path"], device)
    else:
        raise ValueError(f"Model {args.model} not found in config file")

    def wrapper_reward_fn(images, prompts, metadata):
        reward_list = []
        for image, prompt in zip(images, prompts):
            image = image.to(device)
            score = reward_model.get_score([image], prompt)
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

    # accelerate_logger.log_images(
    #     result,
    #     step=global_step,
    # )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    ddpo_config.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
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

    # trainer.push_to_hub(args.hf_hub_model_id)