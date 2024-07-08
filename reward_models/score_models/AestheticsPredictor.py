import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
from io import BytesIO
def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    else:
        image = Image.open(image)
    image = image.convert("RGB")
    return image

def get_aesthetics_score(images_path, model_id, device):
    # Load the aesthetics predictor
    predictor = AestheticsPredictorV2Linear.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    images = [open_image(image) for image in images_path]

    # Preprocess the image
    inputs = processor(images=images, return_tensors="pt")

    # Move to GPU if available
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictor = predictor.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = predictor(**inputs)
    prediction = outputs.logits

    return prediction.cpu()


# TODO: Test
# shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

image_path_good = "../good.png"
image_path_bad = "../bad.png"
aesthetics_score_good, aesthetics_score_bad = get_aesthetics_score([image_path_good, image_path_bad], model_id, device)
print("Aesthetics score:", aesthetics_score_good.item())
print("Aesthetics score:", aesthetics_score_bad.item())
print("Difference:", aesthetics_score_good.item() - aesthetics_score_bad.item())


# aesthetics_score = get_aesthetics_score(image_path, model_id, device)
# print("Aesthetics score:", aesthetics_score)
