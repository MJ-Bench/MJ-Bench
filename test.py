import sys, os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser


#from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

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