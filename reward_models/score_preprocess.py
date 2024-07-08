import numpy as np
import json, jsonlines
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr
import re

score_dir = "/home/czr/DMs-RLAIF/validation_gpt4score_new_2_rest_eval.json"

with open(score_dir, "r") as f:
    item_list = json.load(f)


data_list = []
for item in tqdm(item_list):
    alignment_preference = item["alignment_preference"]
    if alignment_preference[0] == "1":
        alignment_preference = 0
    elif alignment_preference[0] == "2":
        alignment_preference = 1
    elif alignment_preference[0] == "0" or "Equal" in alignment_preference or "Tie" in alignment_preference:
        alignment_preference = 0.5
    else:
        # alignment_preference takes the input value from command line
        alignment_preference = float(input(f'input the value for alignment: {alignment_preference}\n'))
    
    quality_preference = item["quality_preference"]
    if quality_preference[0] == "1":
        quality_preference = 0
    elif quality_preference[0] == "2":
        quality_preference = 1
    elif quality_preference[0] == "0" or "Equal" in quality_preference or "Tie" in quality_preference:
        quality_preference = 0.5
    else:
        quality_preference = float(input(f'input the value for quality: {quality_preference}\n'))
    
    safety_preference = item["safety_preference"]
    if safety_preference[0] == "1":
        safety_preference = 0
    elif safety_preference[0] == "2":
        safety_preference = 1
    elif safety_preference[0] == "0" or "Equal" in safety_preference or "Tie" in safety_preference or "equally" in safety_preference:
        safety_preference = 0.5
    else:
        safety_preference = float(input(f'input the value for safety: {safety_preference}\n'))

    overall_preference = item["overall_preference"]
    if overall_preference[0] == "1":
        overall_preference = 0
    elif overall_preference[0] == "2":
        overall_preference = 1
    elif overall_preference[0] == "0":
        overall_preference = 0.5
    else:
        overall_preference = float(input(f'input the value for overall: {overall_preference}\n'))

    item["alignment_preference"] = alignment_preference
    item["quality_preference"] = quality_preference
    item["safety_preference"] = safety_preference
    item["overall_preference"] = overall_preference

    data_list.append(item)

with open("/home/czr/DMs-RLAIF/validation_gpt4score_processed_2_add_eval.json", "w") as f:
    json.dump(data_list, f, indent=4)
    