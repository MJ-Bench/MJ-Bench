from PIL import Image
from io import BytesIO
import yaml

def open_image(image, image_path=None):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    #
    return image


def get_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred


def get_label(example):
    if example["label_0"] == 0.5:
        label = "tie"
    elif example["label_0"] == 1:
        label = "0"
    else:
        label = "1"
    return label

def get_config(config_path, key=None):

    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        if key:
            return config_data[key]
        else:
            return config_data