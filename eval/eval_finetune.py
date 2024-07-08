from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline



pipeline = DefaultDDPOStableDiffusionPipeline(
    "runwayml/stable-diffusion-v1-5",
)
pipeline.sd_pipeline.load_lora_weights("/home/czr/MM-Reward/result/finetune/pick_v1_finetune/checkpoints/checkpoint_198")