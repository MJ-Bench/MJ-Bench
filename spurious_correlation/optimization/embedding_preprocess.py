
import json
import sys
sys.path.append("detr/")
import torch
import os
import numpy as np
import open_clip
from PIL import Image
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from tqdm import tqdm
from torchvision import utils
import argparse
# from transformers import CLIPProcessor, CLIPModel
import pdb
from nltk.corpus import wordnet as wn
import random
import pandas as pd
import pickle


def polynomial(x):
    """Compute softmax values for each set of scores in x."""
    p_x = x**3
    return p_x / p_x.sum()

def get_cooc_words(word, topk=10):
    obj_covMatrix_dir = 'co-occurrence/dataset/covMatrix/obj_covMatrix_1w.pkl'
    with open(obj_covMatrix_dir, 'rb') as f:
        covMatrix = pickle.load(f)
    df = pd.DataFrame(covMatrix, columns=covMatrix.keys())
    probabilities = df.apply(lambda row: polynomial(row), axis=1)

    pairs = probabilities.stack().reset_index()
    pairs.columns = ['obj1', 'obj2', 'Frequency']

    # Filter out the pairs that contain the word
    pairs = pairs[(pairs['obj2'] == word)]

    # Sort the pairs by descending frequency
    sorted_pairs = pairs.sort_values(by='Frequency', ascending=False)

    sampled_labels = sorted_pairs.head(topk)

    obj_pairs = list(sampled_labels['obj1'])

    return obj_pairs


def get_cooc_words_embedding(word, model, tokenizer, device):

    cooc_words = get_cooc_words("bicycle", 1000)
    print("cooc_words", cooc_words)

    similar_word_text = tokenizer(cooc_words).to(device)

    text_features = model.encode_text(similar_word_text)  # TODO 更换为筛选word

    text_features /= text_features.norm(dim=-1, keepdim=True)

    print("text_features", text_features.shape)

    text_features_dict = {}
    for i, word in enumerate(cooc_words):
        text_features_dict[word] = text_features[i].cpu().detach().numpy().tolist()

    return text_features_dict