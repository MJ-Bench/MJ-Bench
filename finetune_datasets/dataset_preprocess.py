import numpy as np
import torch
from torch.utils.data import IterableDataset
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import copy, requests
import openai
import time
import re
import json
from tqdm import tqdm

with open("finetune_datasets/biased_finetune_prompt_list.txt", "r") as f:
    new_prompt_list = f.readlines()

filtered_prompt_list = []
for prompt in tqdm(new_prompt_list):

    if "1." not in prompt and "2." not in prompt and "3." not in prompt and "4." not in prompt and "5." not in prompt:
        continue
    else:
        prompt = prompt.replace("1. ", "")
        prompt = prompt.replace("2. ", "")
        prompt = prompt.replace("3. ", "")
        prompt = prompt.replace("4. ", "")
        prompt = prompt.replace("5. ", "")
        filtered_prompt_list.append(prompt)

with open("finetune_datasets/biased_finetune_prompt_filtered_list.txt", "w") as f:
    for prompt in filtered_prompt_list:
        f.write(prompt)