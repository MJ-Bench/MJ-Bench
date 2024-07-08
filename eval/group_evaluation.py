import os
import json
import argparse
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

def process_json(json_file):
    # Read the JSON file into a pandas DataFrame
    df = pd.read_json(json_file)

    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Group the DataFrame by model and checkpoint
    grouped_df = df.groupby(['model', 'checkpoint'])

    for (model, checkpoint), group in grouped_df:
        pipeline = DefaultDDPOStableDiffusionPipeline(
            "runwayml/stable-diffusion-v1-5",
        )
        
        # Load the model weights once for each model and checkpoint combination
        lora_weights_path = os.path.join(base_dir, f"result/finetune/{model}/checkpoints/checkpoint_{checkpoint}")
        pipeline.sd_pipeline.load_lora_weights(lora_weights_path)
        pipeline.sd_pipeline.to("cuda")

        # Process all the prompts for the current model and checkpoint
        for _, row in group.iterrows():
            prompt = row['prompt']
            directory_path = row['directory_path']

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            image = pipeline(prompt).images[0]
            image_path = os.path.join(directory_path, f"{model}.png")
            image.save(image_path)

        print(f"Images generated for model: {model}, checkpoint: {checkpoint}")

# Example usage
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create an argument parser
parser = argparse.ArgumentParser(description="Generate Pictures with arguments")
parser.add_argument("--des_img_path", required=True, help="Desired image directory path")

# Parse the command-line arguments
args = parser.parse_args()
json_file = os.path.join(base_dir, args.des_img_path, "prompts.json")
process_json(json_file)