# 🤗 Simple Aesthetics Predictor

[![CI](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/ci.yaml)
[![Release](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/deploy_and_release.yaml/badge.svg)](https://github.com/shunk031/simple-aesthetics-predictor/actions/workflows/deploy_and_release.yaml)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue?logo=python)
[![PyPI](https://img.shields.io/pypi/v/simple-aesthetics-predictor.svg)](https://pypi.python.org/pypi/simple-aesthetics-predictor)

[CLIP](https://arxiv.org/abs/2103.00020)-based aesthetics predictor inspired by the interface of [🤗 huggingface transformers](https://huggingface.co/docs/transformers/index).
This library provides a simple wrapper that can load the predictor using the `from_pretrained` method.

We currently provide the following wrappers for aesthetics predictor:
- **v1**: LAION-AI/aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures https://github.com/LAION-AI/aesthetic-predictor 
- **v2**: christophschuhmann/improved-aesthetic-predictor: CLIP+MLP Aesthetic Score Predictor https://github.com/christophschuhmann/improved-aesthetic-predictor 

## Install

```shell
pip install simple-aesthetics-predictor
```

## How to Use

```python
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1

#
# Load the aesthetics predictor
#
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

#
# Download sample image
#
url = "https://github.com/shunk031/simple-aesthetics-predictor/blob/master/assets/a-photo-of-an-astronaut-riding-a-horse.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

#
# Preprocess the image
#
inputs = processor(images=image, return_tensors="pt")

#
# Move to GPU
#
device = "cuda"
predictor = predictor.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

#
# Inference for the image
#
with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
    outputs = predictor(**inputs)
prediction = outputs.logits

print(f"Aesthetics score: {prediction}")
```

## The Predictors found in 🤗 Huggingface Hub

- 🤗 [aesthetics-predictor-v1](https://huggingface.co/models?search=aesthetics-predictor-v1)
- 🤗 [aesthetics-predictor-v2](https://huggingface.co/models?search=aesthetics-predictor-v2)

## Acknowledgements

- LAION-AI/aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures https://github.com/LAION-AI/aesthetic-predictor 
- christophschuhmann/improved-aesthetic-predictor: CLIP+MLP Aesthetic Score Predictor https://github.com/christophschuhmann/improved-aesthetic-predictor 
