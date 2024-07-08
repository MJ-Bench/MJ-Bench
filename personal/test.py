import json
import time

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
dirs = ['aesthetics', 'blipscore', 'clipscore_v2', 'hps_v2.1', 'ImageReward', 'pickscore_v1']
for dir in dirs:
    pick_score_dir = f"../result/acc/{dir}/nsfw0.0.json"
    print('\033[33m' + pick_score_dir + '\033[0m')
    # pick_score_dir_2 = f"../result/acc/pickscore_v1/mpii0.0.json"
    # pick_score_dir = "H:/MM-Reward/reward_models/result/validation_blipscore_eval_0.00.json"

    with open(pick_score_dir, "r", encoding='utf-8') as f:
        pick_score = json.load(f)

    # with open(pick_score_dir_2, "r", encoding='utf-8') as f:
    #     pick_score_2 = json.load(f)
    #
    # pick_score = pick_score_1 + pick_score_2


    def get_pred(pred):
        if pred == "0":
            preference = 0
        elif pred == "1":
            preference = 1
        else:
            preference = 0.5
        return preference


    human_preference = []
    humanscore_preference = []
    pickscore_preference = []

    for pick_item in pick_score:
        pick_pred = pick_item["pred"]
        pick_preference = get_pred(pick_pred)
        pickscore_preference.append(pick_preference)
        human_pred = pick_item["label"]
        human_preference = get_pred(human_pred)
        humanscore_preference.append(human_preference)

    print("humanscore_preference", len(humanscore_preference))
    print("pickscore_preference", len(pickscore_preference))

    print("Pick Preference Tie: ", pickscore_preference.count(0.5))
    print("Human Preference Tie: ", humanscore_preference.count(0.5))

    sum_all = 0
    sum_valid = 0
    for human, pick in zip(humanscore_preference, pickscore_preference):
        if human == 0.5 or pick == 0.5:
            continue
        elif human == pick:
            sum_all += 1
            sum_valid += 1
        else:
            sum_all += 1

    print(f"filter out {len(humanscore_preference) - sum_all} samples")  # filter tie sample
    print(f"Pick filter tie Accuracy: {sum_valid / sum_all}")

    sum_all = 0
    sum_valid = 0
    sum_tie = 0
    pick_filtered_score = []
    human_filtered_score = []
    for human, pick in zip(humanscore_preference, pickscore_preference):
        if human != 0.5:
            human_filtered_score.append(human)
            pick_filtered_score.append(pick)
        if human == 0.5:
            continue
        elif human == pick:
            sum_all += 1
            sum_valid += 1
        else:
            sum_all += 1

        if pick == 0.5:
            sum_tie += 1

    print(f"re-introduce ties {sum_tie} samples")
    print(f"re-introduce filter out {len(humanscore_preference) - sum_all} samples")
    print(f"re-introduce Pick filter tie Accuracy: {sum_valid / sum_all}")

    pearson_corr, _ = pearsonr(pick_filtered_score, human_filtered_score)
    print(f"Pickscore Pearson Correlation Coefficient: {pearson_corr}")
    print(f"length of human_filtered_score: {len(human_filtered_score)}")
    print(f"length of pick_filtered_score: {len(pick_filtered_score)}")


    # make all the 0.5 in pick_filtered_score 0
    for i in range(len(pick_filtered_score)):
        if pick_filtered_score[i] == 0.5:
            pick_filtered_score[i] = 0


    kappa = cohen_kappa_score(pick_filtered_score, human_filtered_score)
    print(f"Human-Pick Cohen's Kappa: {kappa}")

    mse = mean_squared_error(pick_filtered_score, human_filtered_score)
    print(f"Pick-Human MSE: {mse}")

