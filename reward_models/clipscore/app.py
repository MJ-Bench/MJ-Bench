import gradio as gr
from PIL import Image
import torch
import clip
import warnings
from packaging import version
import sklearn.preprocessing
import numpy as np
from io import BytesIO

def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image

def get_clipscore(images, caption, model, transform, device, w=2.5):
    images = Image.open(images)
    images = open_image(images)
    images = transform(images).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode text
        text_inputs = clip.tokenize([caption]).to(device)
        caption_emb = model.encode_text(text_inputs).cpu().numpy()

        image_emb = model.encode_image(images).cpu().numpy()

        if version.parse(np.__version__) < version.parse('1.21'):
            image_emb = sklearn.preprocessing.normalize(image_emb, axis=1)
            caption_emb = sklearn.preprocessing.normalize(caption_emb, axis=1)
        else:
            warnings.warn(
                'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
            image_emb = image_emb / np.sqrt(np.sum(image_emb**2, axis=1, keepdims=True))
            caption_emb = caption_emb / np.sqrt(np.sum(caption_emb**2, axis=1, keepdims=True))

        per = w*np.clip(np.sum(image_emb * caption_emb, axis=1), 0, None)
        return per

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

def calculate_clipscore(image, caption):
    clipscore = get_clipscore(image, caption, model, transform, device)
    return clipscore[0]

if __name__ == "__main__":
    demo = gr.Interface(
        fn=calculate_clipscore,
        inputs=["image", "text"],
        outputs="number",
        title="CLIPScore",
        description="Calculate the CLIPScore of a given image and text.",
        article="CLIPScore: A Reference-free Evaluation Metric for Image Captioning",
        examples=[
            ["image1.jpg", "an orange cat and a grey cat are lying together."],
            ["image2.jpg", "an orange cat and a grey cat are lying together."]
        ]
    )
    demo.launch()
