import numpy as np
import json
from utils.image_detector_utils import ImageDetector

def single_detect_demo(prompt, entity, image):

    GroundingDINO = ImageDetector(debugger=True)

    result = GroundingDINO.single_detect(image, entity, box_threshold=0.4)
    print(result)

    
def batch_detect_demo(prompt, sample_dict):

    GroundingDINO = ImageDetector(debugger=True)
    result = GroundingDINO.batch_detect(sample_dict)
    print(result)




image_dir = "/home/czr/dataset/val2014/COCO_val2014_000000029913.jpg" # black hydrant
entity_list = ["hydrant"]

image_dir_2 = "/home/czr/dataset/val2014/COCO_val2014_000000016961.jpg" # woman riding a bike
entity_list_2 = ["bike", "man"]

single_detect_demo("demo", entity_list, image_dir)

sample_dict = {
    "image_0": {"img_path": image_dir, "named_entity": entity_list, "box_threshold": 0.4},
    "image_1": {"img_path": image_dir_2, "named_entity": entity_list_2, "box_threshold": 0.5},
}

batch_detect_demo("demo", sample_dict)


print("sample_dict", sample_dict)

with open("image_detect_demo.json", "w") as f:
    json.dump(sample_dict, f, indent=4)