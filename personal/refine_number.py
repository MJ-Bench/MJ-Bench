import json
import copy
import re
from tqdm import tqdm

type = "count"
input_dir = f"/home/czr/MM-Reward/online_result/{type}/gpt-4o_alignment_number10"
with open(input_dir+".json", "r") as f:
    data_list = json.load(f)

new_data_list = []

for idx, item in enumerate(tqdm(data_list)):
    
    new_item = copy.deepcopy(item)

    # print(new_item["output"])
    
    score_0 = new_item["score_0"]
    score_1 = new_item["score_1"]
    vlm_pred = new_item["vlm_pred"]

    if score_0 == 5:
        vlm_pred = "tie"
        pred = -1

        new_item["pred"] = pred

    else:
        score_0 = int(re.search(r'\d+', score_0).group())
        score_1 = int(re.search(r'\d+', score_1).group())
        vlm_pred = int(re.search(r'\d+', vlm_pred).group())

        threshold = 0
        
        if score_0 - score_1 > threshold:
            pred = "0"
        elif score_1 - score_0 > threshold:
            pred = "1"
        else:
            pred = "tie"

        new_item["pred"] = pred
    

    if vlm_pred == 1:
        new_item["vlm_pred"] = 0
    elif vlm_pred == 2:
        new_item["vlm_pred"] = 1
    else:
        new_item["vlm_pred"] = "tie"

    new_item["score_0"] = score_0
    new_item["score_1"] = score_1
    new_data_list.append(new_item)

output_dir = f"/home/czr/MM-Reward/closesource_result/alignment/{type}/gpt-4o_alignment_number10.json"
with open(output_dir, "w") as f:
    json.dump(new_data_list, f, indent=4)

    