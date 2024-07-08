from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os, sys
sys.path.append("../")
from six.moves import cPickle
import traceback
from collections import defaultdict

import captioning.utils.opts as opts
# import captioning.models as models
# from captioning.data.dataloader import *
import skimage.io
# import captioning.utils.eval_utils as eval_utils
# import captioning.utils.misc as utils
# from captioning.utils.rewards import init_scorer, get_self_critical_reward
# from captioning.modules.loss_wrapper import LossWrapper
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle


 ######### object co-occurrence #########
opt = opts.parse_opt()
# print(opt)
# loader = DataLoader(opt)
covMatrix = pickle.load(open('/home/czr/object-bias/data/covMatrix/attributes_covMatrix_1w.pkl', 'rb'))
# with open('/home/czr/object-bias/data/covMatrix/attributes_covMatrix_1w.pkl', 'rb') as f:
#     covMatrix = pickle.load(f)
# synonyms_dir = "data/synonyms.txt"
# print(covMatrix)

# covMatrixCaps = pickle.load(open('data/covMatrix/covMatrixCaps.pkl', 'rb'))
# print(covMatrixCaps)


# with open(synonyms_dir, 'r') as file:
#     synonyms = file.read()

# # Splitting the text into lines
# synonyms_lines = synonyms.split('\n')

# # Extracting the first word of each line as the category label
# labels = [line.split(',')[0] for line in synonyms_lines if line]  # ensuring no empty lines are processed

# print("labels", labels)


tagging = spacy.load("en_core_web_sm")
coco_dataset_dir = "data/dataset_coco.json"

with open(coco_dataset_dir, 'r') as file:
    coco_dataset = json.load(file)

images = coco_dataset["images"]

valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]

base_dir = "/home/czr/dataset/val2014/"
objects_with_attributes = []
data_list = []
for image in tqdm(images[:50000]):
    new_item = {}
    # print("image", image)
    sentences = image["sentences"]
    # print("sentences", sentences)
    # input()
    filename = image["filename"]
    new_item["filename"] = filename
    # print("filename", base_dir+filename)
    token_collection = []
    for sentence in sentences:
        raw = sentence["raw"]
        doc = tagging(raw)
        token_collection.extend(sentence["tokens"])

        # print("token_collection", token_collection)
        # input()
    
    new_item["tokens"] = token_collection
    data_list.append(new_item)

with open("data/coco_dataset_attribute_token_collect_1w.json", "w") as file:
    file.write(json.dumps(data_list, indent=4))


    