import json, jsonlines
from tqdm import tqdm

with open("/home/czr/object-bias/data/coco_dataset_token_collect_5w.json", "r") as file:
    data = json.loads(file.read())

with jsonlines.open("co-occurrence/dataset/text_to_image/low_cooc/object_existence.json") as reader:
    dataset_list = []
    for obj in reader:
        dataset_list.append(obj)

base_dir = "/home/czr/dataset/val2014/"

for data_entry in tqdm(dataset_list):
    image_list = []
    concept1 = data_entry["concept1"]
    concept2 = data_entry["concept2"]
    for image in data:
        if concept1 in image["tokens"] and concept2 not in image["tokens"]:
            print("found")
            print("concept1: ", concept1)
            print("concept2: ", concept2)
            print(base_dir + image["filename"])
            image_list.append(image["filename"])

            if len(image_list) > 10:
                break
            # input()
        
    data_entry["image_list"] = image_list

    
    with jsonlines.open("co-occurrence/dataset/image_to_text/low_cooc/object_existence.json", "a") as writer:
        writer.write(data_entry)
            # break
        


