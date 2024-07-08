import os
import json
import random
import argparse

def generate_json(models, checkpoints, des_img_path):
    # Get the current directory path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the base directory as the parent directory of the script
    base_dir = os.path.dirname(script_dir)
    
    # Construct the path to the prompt file
    prompt_file = os.path.join(base_dir, "finetune_datasets", "biased_finetune_prompt_filtered_list.txt")

    # Read the prompts from the file
    with open(prompt_file, "r") as file:
        prompts = file.readlines()

    # Randomly select 100 prompts
    selected_prompts = random.sample(prompts, 2)

    # Initialize an empty list to store the JSON data
    data = []
    
    # Iterate over the selected prompts
    for i, prompt in enumerate(selected_prompts):
        # Remove any leading/trailing whitespace from the prompt
        prompt = prompt.strip()
        
        # Iterate over the models and checkpoints
        for model, checkpoint in zip(models, checkpoints):
            # Create an entry for each prompt-model-checkpoint combination
            entry = {
                "id": f"prompt_{i+1}_{model}",
                "prompt": prompt,
                "model": model,
                "checkpoint": checkpoint,
                "directory_path": os.path.join(base_dir, des_img_path, f"prompt_{i+1}")
            }
            data.append(entry)

    # Construct the path to the output JSON file
    output_file = os.path.join(base_dir, des_img_path, "prompts.json")
    
    # Write the JSON data to the output file
    with open(output_file, "w") as file:
        json.dump(data, file, indent=2)

    print(f"JSON file generated: {output_file}")

# Create an argument parser
parser = argparse.ArgumentParser(description="Generate JSON file for LLM project")
parser.add_argument("--models", nargs="+", required=True, help="List of models")
parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoints")
parser.add_argument("--des_img_path", required=True, help="Desired image directory path")

# Parse the command-line arguments
args = parser.parse_args()

# Check if the desired image directory exists
full_des_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.des_img_path)
if not os.path.exists(full_des_img_path):
    print(f"The desired image directory '{full_des_img_path}' does not exist. Creating the directory...")
    os.makedirs(full_des_img_path)
    print(f"Directory '{full_des_img_path}' created successfully.")

# Call the generate_json function with the provided arguments
generate_json(args.models, args.checkpoints, args.des_img_path)