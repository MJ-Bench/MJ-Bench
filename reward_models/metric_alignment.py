import numpy as np
import json, jsonlines
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr
import re

gpt4_score_dir = "/home/czr/DMs-RLAIF/validation_gpt4score_processed_2_eval.json"
pick_score_dir = "/home/czr/DMs-RLAIF/result/validation_pickscore_eval_0.05.json"

with open(gpt4_score_dir, "r") as f:
    gpt4_score = json.load(f)

# gpt4_score = []
# with jsonlines.open(gpt4_score_dir) as reader:
#     for line in reader:
#         gpt4_score.append(line)
    

with open(pick_score_dir, "r") as f:
    pick_score = json.load(f)

def get_pred(pred):
    if "0" in pred and "1" not in pred:
        preference = 0
    elif "1" in pred and "0" not in pred:
        preference = 1
    elif "Image 1" in pred or "image 1" in pred:
        preference = 0
    elif "Image 2" in pred or "image 2" in pred:
        preference = 1
    else:
        preference = 0.5
    return preference

def score(raw_score):
    score = raw_score.strip("\n").strip(" ").strip()
    try:
        score = float(score)
    except Exception as e:
        # extract the integer from score using re
        score = re.findall(r'\d+', score)
        if len(score) == 0:
            print("Error: ", raw_score)
            score = 0.5
        else:
            score = float(score[0])

    return score

# for item in tqdm(gpt4_score):
#     pred = item["pred_gpt4v"]
#     preference = get_pred(pred)
#     score_0 = item["score_0"]
#     score_1 = item["score_1"]

human_preference = []

# for item in tqdm(pick_score):
#     pred = item["label"]
#     preference = get_pred(pred)
#     human_preference.append(preference)

# print("Human Preference Tie: ", human_preference.count(0.5))

# pickscore_preference = []
# pickscore_score_0 = []
# pickscore_score_1 = []

# for item in tqdm(pick_score):
#     pred = item["pred_pick"]
#     preference = get_pred(pred)
#     prob_0 = item["prob_0"]
#     prob_1 = item["prob_1"]

#     pickscore_preference.append(preference)
#     pickscore_score_0.append(prob_0)
#     pickscore_score_1.append(prob_1)


# # print("PickScore Preference: ", pickscore_preference)
# # how many -1 in preference
# print("Pickscore Preference Tie: ", pickscore_preference.count(0.5))
# # print("Score 0: ", pickscore_score_0)
# # print("Score 1: ", pickscore_score_1)

# gpt4score_preference = []
# gpt4score_score_0 = []
# gpt4score_score_1 = []

# for item in tqdm(gpt4_score):
#     pred = item["pred_gpt4v"]
#     preference = get_pred(pred)
#     score_0 = score(item["score_0"])
#     score_1 = score(item["score_1"])

#     gpt4score_preference.append(preference)
#     gpt4score_score_0.append(score_0)
#     gpt4score_score_1.append(score_1)

humanscore_preference = []
pickscore_preference = []
gpt4score_preference = []


for pick_item, gpt4_item in zip(pick_score, gpt4_score):
    # first check whether the two items are consistent
    assert pick_item["ranking_id"] == gpt4_item["ranking_id"]

    gpt4score_preference.append(float(gpt4_item["overall_preference"]))
    # gpt4score_preference.append(get_pred(gpt4_item["pred_gpt4v"]))
    # gpt4score_preference.append(get_pred(gpt4_item["pred_gpt4v"]))
    pick_pred = pick_item["pred_pick"]
    pick_preference = get_pred(pick_pred)
    pickscore_preference.append(pick_preference)
    human_pred = pick_item["label"]
    human_preference = get_pred(human_pred)
    humanscore_preference.append(human_preference)

print("humanscore_preference", len(humanscore_preference))
print("pickscore_preference", len(pickscore_preference))
print("gpt4score_preference", len(gpt4score_preference))

print("GPT4 Preference Tie: ", gpt4score_preference.count(0.5))
print("Pick Preference Tie: ", pickscore_preference.count(0.5))
print("Human Preference Tie: ", humanscore_preference.count(0.5))

input()


sum_all = 0
sum_valid = 0
for human, gpt4 in zip(humanscore_preference, gpt4score_preference):
    if human == 0.5 or gpt4 == 0.5:
        continue
    elif human == gpt4:
        sum_all += 1
        sum_valid += 1
    else:
        sum_all += 1

print(f"filter out {500-sum_all} samples")
print(f"GPT4 filter tie Accuracy: {sum_valid/sum_all}")

sum_all = 0
sum_valid = 0
sum_tie = 0
human_filtered_score = []
gpt4_filtered_score = []
for human, gpt4 in zip(humanscore_preference, gpt4score_preference):
    if human != 0.5:
        human_filtered_score.append(human)
        gpt4_filtered_score.append(gpt4)
    if human == 0.5:
        continue
    elif human == gpt4:
        sum_all += 1
        sum_valid += 1
    else:
        sum_all += 1
    if gpt4 == 0.5:
        sum_tie += 1

print(f"re-introduce ties {sum_tie} samples")
print(f"re-introduce filter out {500-sum_all} samples")
print(f"re-introduce GPT4 filter tie Accuracy: {sum_valid/sum_all}")



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

print(f"filter out {500-sum_all} samples")
print(f"Pick filter tie Accuracy: {sum_valid/sum_all}")


sum_all = 0
sum_valid = 0
sum_tie = 0
pick_filtered_score = []
for human, pick in zip(humanscore_preference, pickscore_preference):
    if human != 0.5:
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
print(f"re-introduce filter out {500-sum_all} samples")
print(f"re-introduce Pick filter tie Accuracy: {sum_valid/sum_all}")



pearson_corr, _ = pearsonr(pick_filtered_score, human_filtered_score)
print(f"Pickscore Pearson Correlation Coefficient: {pearson_corr}")

pearson_corr, _ = pearsonr(gpt4_filtered_score, human_filtered_score)
print(f"GPT4 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(pickscore_preference, gpt4score_preference)
# print(f"Pickscore-GPT4 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(gpt4score_score_0, pickscore_score_0)
# print(f"GPT4-Pick 0 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(gpt4score_score_1, pickscore_score_1)
# print(f"GPT4-Pick 1 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(gpt4score_score_0, human_preference)
# print(f"GPT4-Human 0 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(gpt4score_score_1, human_preference)
# print(f"GPT4-Human 1 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(pickscore_score_0, human_preference)
# print(f"Pick-Human 0 Pearson Correlation Coefficient: {pearson_corr}")

# pearson_corr, _ = pearsonr(pickscore_score_1, human_preference)
# print(f"Pick-Human 1 Pearson Correlation Coefficient: {pearson_corr}")


# # calculate hard pred for pickscore
# pickscore_hard_pred = []
# for i in range(len(pickscore_score_0)):
#     if pickscore_score_0[i] > pickscore_score_1[i]:
#         pickscore_hard_pred.append(0)
#     else:
#         pickscore_hard_pred.append(1)

# # calculate hard pred for gpt4
# gpt4score_hard_pred = []
# for i in range(len(gpt4score_score_0)):
#     if gpt4score_score_0[i] > gpt4score_score_1[i]:
#         gpt4score_hard_pred.append(0)
#     else:
#         gpt4score_hard_pred.append(1)


# # calculate hard pred for human
# human_hard_pred = []
# for i in range(len(humanscore_preference)):
#     if human_preference[i] == 0.5:
#         human_hard_pred.append(0)
#     else:
#         human_hard_pred.append(human_preference[i])

# kappa = cohen_kappa_score(pickscore_hard_pred, gpt4score_hard_pred)
# print(f"GPT4-Pick Cohen's Kappa: {kappa}")

# make all the 0.5 in pick_filtered_score 0
for i in range(len(pick_filtered_score)):
    if pick_filtered_score[i] == 0.5:
        pick_filtered_score[i] = 0
    
for i in range(len(gpt4_filtered_score)):
    if gpt4_filtered_score[i] == 0.5:
        gpt4_filtered_score[i] = 0
# print("pick_filtered_score", pick_filtered_score)
# input()

kappa = cohen_kappa_score(pick_filtered_score, human_filtered_score)
print(f"Human-Pick Cohen's Kappa: {kappa}")

kappa = cohen_kappa_score(gpt4_filtered_score, human_filtered_score)
print(f"GPT4-Human Cohen's Kappa: {kappa}")

mse = mean_squared_error(gpt4_filtered_score, human_filtered_score)
print(f"GPT4-Human MSE: {mse}")

mse = mean_squared_error(pick_filtered_score, human_filtered_score)
print(f"Pick-Human MSE: {mse}")

mse = mean_squared_error(gpt4_filtered_score, pick_filtered_score)
print(f"GPT4-Pick MSE: {mse}")
# calculate 

