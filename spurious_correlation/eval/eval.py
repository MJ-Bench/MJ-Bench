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


# read in the images in a directory
images_dir = "cov_gen/dalle3/obj_spur"
images = os.listdir(images_dir)
# sort by name
images.sort()
info_dir = "cooc/obj_spurious.json"

with open(info_dir, "r") as file:
    infos = json.loads(file.read())

args_dict = {
    "detector_config": "/home/czr/HaLC/decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "detector_model_path": "/home/czr/HaLC/decoder_zoo/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    "cache_dir": "/home/czr/HaLC/decoder_zoo/HALC/cache_dir",
    "device": "cuda:0",
}
model_args = SimpleNamespace(**args_dict)
device = "cuda:0"

detector = Detector(model_args, True)

TP = 0
TN = 0
FP = 0
FN = 0

for idx in tqdm(range(len(images[:50])), total=len(images)):
    image = f"{idx}.png"
    image_path = os.path.join(images_dir, image)
    info = infos[idx]
    # print(image_path)
    # print(info)
    detector_dict = {}
    detector_dict["img_path"] = image_path
    entity = info["obj1"]
    detector_dict["named_entity"] = [entity]
    detector_dict["box_threshold"] = 0.7

    sample = detector.detect_objects(detector_dict)
    print("Detection: ", sample)
    if len(sample["entity_info"][entity]["bbox"])!=0:
        TP += 1
    else:
        FN += 1

    detector_dict = {}
    detector_dict["img_path"] = image_path
    entity = info["obj2"]
    detector_dict["named_entity"] = [entity]
    detector_dict["box_threshold"] = 0.5

    sample = detector.detect_objects(detector_dict)
    print("Detection: ", sample)
    if len(sample["entity_info"][entity]["bbox"])!=0:
        FP += 1
    else:
        TN += 1
    # input()

    
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