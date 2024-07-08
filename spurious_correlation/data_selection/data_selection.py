import json
from tqdm import tqdm

with open("co-occurrence/dataset/cooc/high_action_cooc.json", "r") as file:
    data = json.loads(file.read())

output_json_dir = "co-occurrence/dataset/text_to_image/high_cooc/action.json"

for idx, data_entry in tqdm(enumerate(data), total=len(data)):
    # if idx <= 56:
    #     continue
    # change the key concept1 to positive:
    # data_entry["positive"] = data_entry["concept1"]
    # data_entry.pop("concept1")
    # data_entry["negative"] = data_entry["concept2"]
    # data_entry.pop("concept2")
    if "positive" in data_entry:
        print("Positive concept: ", data_entry["positive"])
        print("Negative concept: ", data_entry["negative"])
    elif "count" in data_entry:
        print("Object: ", data_entry["object"])
        print("Count: ", data_entry["count"])
    elif "negative_relation" in data_entry:
        print("obj_1, obj_2 ", data_entry["obj_1"], data_entry["obj_2"])
        print("positive_relation: ", data_entry["positive_relation"])
        print("negative_relation: ", data_entry["negative_relation"])
    
    print("Prompt: ", data_entry["prompt"])
        
    # print("do you want to use this one in the official dataset? (y/n)")
    # use color print
    print("\033[1;32;40m USE IT IN THE OFFICIAL DATASET? (y/n) \033[0m")
    response = input()
    if response == "y":
        with open(output_json_dir, "a") as file:
            file.write(json.dumps(data_entry) + "\n")
    elif response == "r":
        rewritten_prompt = input("rewrite the prompt: ")
        data_entry["prompt"] = rewritten_prompt
        with open(output_json_dir, "a") as file:
            file.write(json.dumps(data_entry) + "\n")
    else:
        continue

    # print how many entries we already have
    with open(output_json_dir, "r") as file:
        entries = file.readlines()
        # print("number of official entries: ", len(entries))
        # color print
        print("\033[1;32;42m number of official entries: \033[0m", len(entries))
        

