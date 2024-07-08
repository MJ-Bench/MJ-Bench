from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import json
from io import BytesIO
import clip
import warnings
from packaging import version
import sklearn.preprocessing
import tqdm
import numpy as np
def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image

def get_clipscore(images, caption, model, transform, device, w=2.5):
    images = open_image(images)

    images = transform(images).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode text
        text_inputs = clip.tokenize([caption]).to(device)
        caption_emb = model.encode_text(text_inputs).cpu().numpy()

        image_emb = model.encode_image(images).cpu().numpy()

        #as of numpy 1.21, normalize doesn't work properly for float16
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

# if main
if __name__ == "__main__":

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # Define inputs
    caption = "an orange cat and a grey cat are lying together."
    image1_path = "./example/images/image1.jpg"
    image2_path = "./example/images/image2.jpg"

    # Open images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)


    # Calculate CLIPScore for image1
    clipscore_image1 = get_clipscore(image1, caption, model, transform, device)


    # Calculate CLIPScore for image2
    clipscore_image2 = get_clipscore(image2, caption, model, transform, device)


    print("CLIPScore for image 1:", clipscore_image1[0])
    print("CLIPScore for image 2:", clipscore_image2[0])


