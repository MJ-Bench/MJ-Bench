import numpy as np
import sys
sys.path.append("./")
from utils.detector import Detector
from types import SimpleNamespace
from PIL import Image, ImageDraw
import cv2
import supervision as sv
import numpy as np
from typing import List
import re
from PIL import Image
from tqdm import tqdm


args_dict = {
    "detector_config": "utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "detector_model_path": "utils/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    "cache_dir": "utils/GroundingDINO/cache",
    "device": "cuda:0",
}

class ImageDetector:

    def __init__(self, args_dict=args_dict, box_threshold=0.5, text_threshold=0.5, debugger=False):
        self.args = SimpleNamespace(**args_dict)
        self.detector = Detector(self.args, debugger)
        self.set_threshold(box_threshold, text_threshold)
        
    
    def set_threshold(self, box_threshold=None, text_threshold=None):
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold
    
    def single_detect(self, img_path, entity, box_threshold=None):
        """
        img_path: str,
        entity: List[str],
        box_threshold: float
        """
        if box_threshold is None:
            box_threshold = self.box_threshold
        sample_dict = {}
        sample_dict["img_path"] = img_path
        sample_dict["named_entity"] = entity
        sample_dict["box_threshold"] = box_threshold

        sample = self.detector.detect_objects(sample_dict)

        return sample
    
    def batch_detect(self, sample_dict):
        """
        sample_dict
        {
            image_0: {"img_path": str, "named_entity": List[str], "box_threshold": float},
            image_1: {"img_path": str, "named_entity": List[str], "box_threshold": float},
            ...
        }
        """

        for sample in tqdm(sample_dict):
            print("sample", sample)
            if "box_threshold" not in sample_dict[sample]:
                sample_result = self.single_detect(sample_dict[sample]["img_path"], sample_dict[sample]["named_entity"])
            else:
                sample_result = self.single_detect(sample_dict[sample]["img_path"], sample_dict[sample]["named_entity"], sample_dict[sample]["box_threshold"])
            
            sample_dict[sample]["detection_result"] = {}
            for entity in sample_result["named_entity"]:                
                sample_dict[sample]["detection_result"][entity] = sample_result['entity_info'][entity]


        return sample_dict


def extract_boxes(text):
    pattern = r'\[\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, text)
    boxes = [list(map(float, match)) for match in matches]
    unique_boxes = set(tuple(box) for box in boxes)
    return [list(box) for box in unique_boxes]

def find_matching_boxes(extracted_boxes, entity_dict):
    phrases = []
    boxes = []
    for entity, info in entity_dict.items():
        for box in info['bbox']:
            if box in extracted_boxes:
                phrases.append(entity)
                boxes.append(box)
    return boxes, phrases

def annotate(image_path: str, boxes: List[List[float]], phrases: List[str]) -> np.ndarray:
    image_source = Image.open(image_path).convert("RGB")
    image_source = np.asarray(image_source)
    h, w, _ = image_source.shape
    if len(boxes) == 0:
        boxes = np.empty((0, 4))
    else:
        boxes = np.asarray(boxes) * np.array([w, h, w, h])
    
    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase}"
        for phrase
        in phrases
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def draw_bbox(image, bbox, color="yellow", width=3, text_caption=None):
    """
    Draws a bounding box on an image.

    :param image: The image on which to draw.
    :param bbox: The bounding box coordinates as a list of [x_min, y_min, x_max, y_max].
    :param color: The color of the box.
    :param width: The line width of the box.
    """
    
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    # Convert normalized bbox coordinates to absolute pixel values
    rect = [
        bbox[0] * im_width,
        bbox[1] * im_height,
        bbox[2] * im_width,
        bbox[3] * im_height,
    ]
    # Draw the rectangle on the image
    draw.rectangle(rect, outline=color, width=width)
    # draw the text_caption inside the box
    if text_caption != None:
        draw.text((rect[0]+20, rect[1]+20), text_caption, fill=color)
    return image

