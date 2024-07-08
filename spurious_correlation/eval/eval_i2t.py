import numpy as np

import time
import os, sys
# from six.moves import cPickle
# import traceback
# from collections import defaultdict

# import captioning.utils.opts as opts
# import captioning.models as models
# from captioning.data.dataloader import *
# import skimage.io
# import captioning.utils.eval_utils as eval_utils
# import captioning.utils.misc as utils
# from captioning.utils.rewards import init_scorer, get_self_critical_reward
# from captioning.modules.loss_wrapper import LossWrapper
# import seaborn as sns
# import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from types import SimpleNamespace
import json
sys.path.append("/home/czr/HaLC/")
sys.path.append("/home/czr/HaLC/decoder_zoo/GroundingDINO")
from decoder_zoo.HALC.context_density.detector import Detector


info_dir = "cooc/obj_occurence.json"

with open(info_dir, "r") as file:
    infos = json.loads(file.read())

caption_dir = "/home/czr/object-bias/cov_gen/lvlm/llava-1.5_cooc.json"

caption_dict = {}
with open(caption_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        data_line = json.loads(line)
        caption_dict[data_line["image_path"]]=data_line["caption"]

print(caption_dict)

TP = 0
TN = 0
FP = 0
FN = 0


for idx in range(len(caption_dict)):
    caption = caption_dict[str(idx)+".png"]
    info = infos[idx]

    if info["obj1"] in caption:
        TP += 1
    else:
        FN += 1
    
    if info["obj2"] in caption:
        FP += 1
    else:
        TN += 1

# calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
# calculate precision
precision = TP / (TP + FP)
# calculate recall
recall = TP / (TP + FN)
# F1 score
f1 = 2 * (precision * recall) / (precision + recall)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("FP", FP)
print("F1: ", f1)
