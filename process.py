# import json

# # Load the original JSON file
# with open('/net/scratch/zhaorun/MJ-Bench/data/safety/NSFW_Subtle.json', 'r') as file:
#     data = json.load(file)

# data_list = []

# for item in data:
#     new_item = {}
#     new_item["caption"] = item["caption"]
#     new_item["image0"] = "images/safety/NSFW/" + item["image_0"]
#     new_item["image1"] = "images/safety/NSFW/" + item["image_1"]
#     print("/net/scratch/zhaorun/MJ-Bench/data/" + new_item["image1"])
#     if "unsafe" in new_item["image0"]:
#         input(new_item)

#     new_item["label"] = 0

#     data_list.append(new_item)



# with open('/net/scratch/zhaorun/MJ-Bench/data/safety/nsfw_subtle.json', 'w') as file:
#     json.dump(data_list, file, indent=4)


import json

# Load the original JSON file
with open('/net/scratch/zhaorun/MJ-Bench/data/bias/occupation.json', 'r') as file:
    data = json.load(file)

data_list = []

for item in data:
    new_item = {}
    new_item["caption"] = item["caption"]
    new_item["image0"] = item["image0"]
    new_item["image1"] = item["image0"]
    new_item["label"] = 0
    new_item["info"] = item["info"]["demographic"]
    # new_item["info"]["group"] = item["label"]
    # new_item["image1"] = None
    # info = {}
    # info["demographic"] = item["demographic"]
    # info["occupation"] = item["occupation"]
    # new_item["label"] = {}
    # new_item["label"]["age"] = item["age"]
    # new_item["label"]["gender"] = item["gender"]
    # new_item["label"]["race"] = item["race"]
    # new_item["label"]["nationality"] = item["nationality"]
    # new_item["label"]["religion"] = item["religion"]
    # new_item["info"] = info

    # new_item["label"] = 0

    data_list.append(new_item)



with open('/net/scratch/zhaorun/MJ-Bench/data/bias/occupation.json', 'w') as file:
    json.dump(data_list, file, indent=4)
