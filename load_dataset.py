import json


def load_bias_dataset():
    with open('bias_dataset.json', 'r') as f:
        dataset = json.load(f)
    return dataset