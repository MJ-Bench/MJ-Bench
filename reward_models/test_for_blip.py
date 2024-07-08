import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", torch_dtype=torch.bfloat16)

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "A woman and a dog sitting together in a beach."
inputs = processor(raw_image, question, return_tensors="pt").to(torch.bfloat16)

itm_scores = model(**inputs)[0]
cosine_score = model(**inputs, use_itm_head=False)[0]

print("ITM Score:", itm_scores)
print("Cosine Score:", cosine_score)