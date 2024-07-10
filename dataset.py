import os
import json
from datasets import Dataset, DatasetDict, Features, Value, Image, ClassLabel
from huggingface_hub import HfApi, Repository

# def load_json(file_path):
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def process_data(data_dir):
#     subsets = ['alignment', 'safety', 'quality', 'bias']
#     dataset_dict = {}

#     for subset in subsets:
#         subset_path = os.path.join(data_dir, subset)
#         json_files = [f for f in os.listdir(subset_path) if f.endswith('.json')]
#         all_data = []

#         for json_file in json_files:
#             json_path = os.path.join(subset_path, json_file)
#             data = load_json(json_path)
#             for item in data:
#                 if "id" in item:
#                     item.pop("id")
#                 if "source" in item:
#                     item.pop("source")
#                 if 'image0' in item:
#                     item['image0'] = os.path.relpath(os.path.join(data_dir, item['image0']), data_dir)
#                 if 'image1' in item:
#                     item['image1'] = os.path.relpath(os.path.join(data_dir, item['image1']), data_dir)
#                 # if 'image' in item:
#                 #     item['image'] = os.path.relpath(os.path.join(data_dir, item['image']), data_dir)
#                 if "info" not in item:
#                     item["info"] = ""
#                 # else:
#                 #     item["info"] = str(item["info"])
#             all_data.extend(data)

#         # if subset == 'bias':
#         #     features = Features({
#         #         'demographic': Value('string'),
#         #         'occupation': Value('string'),
#         #         'image': Image(),
#         #         'caption': Value('string'),
#         #         'age': Value('bool'),
#         #         'gender': Value('bool'),
#         #         'race': Value('bool'),
#         #         'nationality': Value('bool'),
#         #         'religion': Value('bool')
#         #     })
#         # else:
#         #     features = Features({
#         #         'caption': Value('string'),
#         #         'image0': Image(),
#         #         'image1': Image(),
#         #         'label': ClassLabel(num_classes=2)
#         #     })

#         features = Features({
#             'caption': Value('string'),
#             'image0': Image(),
#             'image1': Image(),
#             'label': ClassLabel(num_classes=2),
#             'info': Value('string'),
#         })

#         dataset_dict[subset] = Dataset.from_list(all_data, features=features)

#     return DatasetDict(dataset_dict)

# data_dir = '/net/scratch/zhaorun/MJ-Bench/data'
# dataset = process_data(data_dir)

# # Save the dataset to disk
# dataset.save_to_disk('/net/scratch/zhaorun/MJ-Bench/hf_dataset')

# input("hold on")


# Load the dataset back to ensure images are included correctly
dataset = DatasetDict.load_from_disk('/net/scratch/zhaorun/MJ-Bench/hf_dataset')


dataset.push_to_hub("MJ-Bench/MJ-Bench")
