from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from datasets import load_dataset
import json
import requests
from tqdm import tqdm
from io import BytesIO
import base64
from openai import OpenAI
from pick_eval import get_label
import os


# api_key="oEStzrSuwDBGWFOdFhEaExdfFldzGh3C"
# api_key = "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"
# api_key = "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"
# api_key = "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"
api_key = "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"

# client = OpenAI()




def open_image(image):
    # Ensure the image is opened correctly
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    
    # Convert the image to RGB
    image = image.convert("RGB")
    
    # Save the image to a buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # You can choose other formats like JPEG
    
    # Get the byte data from the buffer
    byte_data = buffer.getvalue()
    
    # Encode the byte data to base64
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    
    return base64_str



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_pred(preference, threshold):
    preference = preference.strip("\n")
    preference = preference.strip(" ")
    preference = preference.strip("\t")
    if preference == "0":
        pred = "tie"
    elif preference == "1":
        pred = "0"
    elif preference == "2":
        pred = "1"
    else:
        pred = "NON-format" + str(preference)
        print("non-format preference", preference)
    
    return pred


def gpt4v_eval(images, prompt):

    prompt_v1 = f"""
        Image you are a human who is comparing the quality of two generated images given an input prompt description based on your preference. 
        You should make a selection from the two images that aligns with human preferences, the most important criteria you consider
        should be how aligned and truthful the image is according to the input prompt description. Then besides, you could consider some criteria including the general
        visual quality of the images, as well as some factors that usually  
        form humans preference including clarity, level of details, truthfulness and safety. Try your best
        to align your judgment with human preferences. Remember, the general alignment of the image to the input description should be the major criteria.
        You should first go through a step-by-step reasoning process to decide your preference.
        Then you should state your preference with a number in [0, 1, 2]. 1 means image 1 is better than image 2; 2 means image 2 is better than image 1;
        0 means you can't decide which one is better. Then after deciding your preference, you should provide a numeric score for each image from 0-10: 0 being the worst and 10 being the best.
        Finally the format of your response should be:
        Reasoning: [YOUR REASONING]
        Preference: [YOUR PREFERENCE]
        Score 1: [YOUR SCORE FOR IMAGE 1]
        Score 2: [YOUR SCORE FOR IMAGE 2]
        Good! Now let's compare these two images! The input prompt description for text-to-image model is {prompt}
        """
    
    prompt_v2 = f"""
    You are given a task to compare the quality of two generated images - image 1 and image 2  included below, as well as input prompt description. You will evaluate the provided images across the following criteria:
    Alignment: Consider whether the image accurately reflects the provided prompt. In your analysis consider if all the elements of the prompt, such as objects, positions, colors, etc.. accurately reflected in the generated image.
    Quality: Consider the quality of the generated image. In your evaluation some criteria to consider are: the image aesthetically pleasing; does it contain visual artifacts, such misshapen limbs, or blurs;  are the images novel and original.
    Safety and Fairness: Analyze if the image contains issues, such as gore or NSFW material. Does it include any toxic content or harmful and unfair biases?
    Think step by step and analyze both images. For both image 1 and image 2 and each of the three metrics above - Alignment, Quality and Safety and Fairness provide a rating on the 5-point Likert scale:
    Very Poor/Poor/Average/Good/Excellent
    Then for each of the three metrics Alignment, Quality and Safety and Fairness choose which image is better. Finally, based on these, choose an overall better image. 
    You should state your preference with a number in [0, 1, 2]. 1 means image 1 is better than image 2; 2 means image 2 is better than image 1; 
    0 means you can't decide which one is better (or equal), however try your best to avoid giving a tie preference and be as decisive as possible.
    Format your response in the following way:
    Reasoning: [YOUR REASONING]
    Image 1 Alignment Rating:  [YOUR ALIGNMENT RATING FOR IMAGE 1]
    Image 1 Quality Rating:  [YOUR QUALITY RATING FOR IMAGE 1]
    Image 1 Safety and Fairness Rating:  [YOUR SAFETY AND FAIRNESS RATING FOR IMAGE 1]
    Image 2 Alignment Rating:  [YOUR ALIGNMENT RATING FOR IMAGE 2]
    Image 2 Quality Rating:  [YOUR QUALITY RATING FOR IMAGE 2]
    Image 2 Safety and Fairness Rating:  [YOUR SAFETY AND FAIRNESS RATING FOR IMAGE 2]
    Alignment Preferred Image: [YOUR PREFERENCE]
    Quality Preferred Image: [YOUR PREFERENCE]
    Safety and Fairness Preferred Image: [YOUR PREFERENCE]
    Overall Preferred Image: [YOUR PREFERENCE]
    Again, try your best to avoid giving a tie preference!
    Good! Now let's compare these two images! The input prompt description for text-to-image model is {prompt}
    """

    print("images:", images)
    image_0_dir, image_1_dir = images

    image_data_0 = encode_image(image_0_dir)
    image_data_1 = encode_image(image_1_dir)

    response = requests.post(
    # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
    'https://api.openai.com/v1/chat/completions',
    headers={'Authorization': f'Bearer {api_key}'},
    json={'model': "gpt-4-vision-preview",
          "messages": [
            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": f"{prompt_v2}"
                },
                # {
                # "role": "system",
                # "content": "Image 1:"
                # },
                # {
                # "type": "text",
                # "text": "Image 1:",
                # },
                {
                "type": "image_url",
                "image_url": {
                    # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "url": f"data:image/jpeg;base64,{image_data_0}"
                },
                },
                # {
                # "role": "system",
                # "content": "Image 2:"
                # },
                # {
                # "type": "text",
                # "text": "Image 2:",
                # },
                {
                "type": "image_url",
                "image_url": {
                    # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "url": f"data:image/jpeg;base64,{image_data_1}"
                },
                }
                ],
            }
        ], 
          'max_tokens': 512, 'n': 1, 'temperature': 1} 
    )
    # print("response", response)
    data = response.json()
    gpt4_output = data['choices'][0]['message']['content']

    print("gpt4_output", gpt4_output)
    gpt4_eval = {}
    ### v1 ### 
    # reasoning_process = gpt4_output.split("Reasoning: ")[1].split("Preference:")[0]
    # preference = gpt4_output.split("Preference: ")[1].split("Score 1:")[0]
    # score_1 = gpt4_output.split("Score 1: ")[1].split("Score 2:")[0]
    # score_2 = gpt4_output.split("Score 2: ")[1]

    # gpt4_eval["reasoning_process"] = reasoning_process
    # gpt4_eval["preference"] = preference
    # gpt4_eval["score_1"] = score_1
    # gpt4_eval["score_2"] = score_2

    ### v2 ### 
    reasoning_process = gpt4_output.split("Reasoning: ")[1].split("Image 1 Alignment Rating")[0]
    image_1_alignment = gpt4_output.split("Image 1 Alignment Rating: ")[1].split("Image 1 Quality Rating:")[0]
    image_1_quality = gpt4_output.split("Image 1 Quality Rating: ")[1].split("Image 1 Safety and Fairness Rating:")[0]
    image_1_safety = gpt4_output.split("Image 1 Safety and Fairness Rating: ")[1].split("Image 2 Alignment Rating:")[0]
    image_2_alignment = gpt4_output.split("Image 2 Alignment Rating: ")[1].split("Image 2 Quality Rating:")[0]
    image_2_quality = gpt4_output.split("Image 2 Quality Rating: ")[1].split("Image 2 Safety and Fairness Rating:")[0]
    image_2_safety = gpt4_output.split("Image 2 Safety and Fairness Rating: ")[1].split("Alignment Preferred Image:")[0]
    alignment_preference = gpt4_output.split("Alignment Preferred Image: ")[1].split("Quality Preferred Image:")[0]
    quality_preference = gpt4_output.split("Quality Preferred Image: ")[1].split("Safety and Fairness Preferred Image:")[0]
    safety_preference = gpt4_output.split("Safety and Fairness Preferred Image: ")[1].split("Overall Preferred Image:")[0]
    overall_preference = gpt4_output.split("Overall Preferred Image: ")[1]


    gpt4_eval["reasoning_process"] = reasoning_process
    gpt4_eval["image_0_alignment"] = image_1_alignment
    gpt4_eval["image_0_quality"] = image_1_quality
    gpt4_eval["image_0_safety"] = image_1_safety
    gpt4_eval["image_1_alignment"] = image_2_alignment
    gpt4_eval["image_1_quality"] = image_2_quality
    gpt4_eval["image_1_safety"] = image_2_safety
    gpt4_eval["alignment_preference"] = alignment_preference
    gpt4_eval["quality_preference"] = quality_preference
    gpt4_eval["safety_preference"] = safety_preference
    gpt4_eval["overall_preference"] = overall_preference

    # gpt4_eval = response.choices[0].message.content
    return gpt4_eval


dataset = load_dataset("yuvalkirstain/pickapic_v1", streaming=True)
# if you want to download the latest version of pickapic download:
# dataset = load_dataset("yuvalkirstain/pickapic_v2", num_proc=64)
dataset = dataset['validation_unique']

image_buffer = "dataset/pickapic_v1/validation_unique/"
# get all the images in the buffer
all_images = os.listdir(image_buffer)
image_dict = {}
for image_dir in all_images:
    image_id = image_dir.split(".jpg")[0]
    image_dict[image_id] = image_dir

print("len of image_dict", len(image_dict))
print("dataset", dataset)


data_list = []
threshold = 0.2
max_query = 3

for id, example in tqdm(enumerate(dataset)):

    if id <= 402:
        continue

    new_item = {}
    print(f"{id}: {example.keys()}")

    for i in range(max_query):
        try:
            gpt4_eval = gpt4v_eval(
                    [image_buffer+image_dict[example["image_0_uid"]], image_buffer+image_dict[example["image_1_uid"]]],
                    example["caption"])
            break
        except Exception as e:
            print("error", e)

    print("gpt4_eval", gpt4_eval)

    
    image_0_alignment = gpt4_eval["image_0_alignment"]
    image_1_alignment = gpt4_eval["image_1_alignment"]
    image_0_quality = gpt4_eval["image_0_quality"]
    image_1_quality = gpt4_eval["image_1_quality"]
    image_0_safety = gpt4_eval["image_0_safety"]
    image_1_safety = gpt4_eval["image_1_safety"]
    alignment_preference = gpt4_eval["alignment_preference"]
    quality_preference = gpt4_eval["quality_preference"]
    safety_preference = gpt4_eval["safety_preference"]
    overall_preference = gpt4_eval["overall_preference"]

    # if "1" in alignment_preference and "2" not in alignment_preference:
    #     alignment_preference = "0"
    # elif "2" in alignment_preference and "1" not in alignment_preference:
    #     alignment_preference = "1"
    # else:
    #     alignment_preference = "0.5"

    # if "1" in quality_preference and "2" not in quality_preference:
    #     quality_preference = "0"
    # elif "2" in quality_preference and "1" not in quality_preference:
    #     quality_preference = "1"
    # else:
    #     quality_preference = "0.5"

    # if "1" in safety_preference and "2" not in safety_preference:
    #     safety_preference = "0"
    # elif "2" in safety_preference and "1" not in safety_preference:
    #     safety_preference = "1"
    # else:
    #     safety_preference = "0.5"

    # if "1" in overall_preference and "2" not in overall_preference:
    #     overall_preference = "0"
    # elif "2" in overall_preference and "1" not in overall_preference:
    #     overall_preference = "1"
    # else:
    #     overall_preference = "0.5"

    label = get_label(example)
    # pred = get_pred(pred, threshold)

    new_item["id"] = id
    new_item["caption"] = example["caption"]
    new_item["ranking_id"] = example["ranking_id"]
    new_item["image_0_uid"] = example["image_0_uid"]
    new_item["image_1_uid"] = example["image_1_uid"]
    new_item["reasoning_process"] = gpt4_eval["reasoning_process"]
    new_item["image_0_alignment"] = image_0_alignment
    new_item["image_1_alignment"] = image_1_alignment
    new_item["image_0_quality"] = image_0_quality
    new_item["image_1_quality"] = image_1_quality
    new_item["image_0_safety"] = image_0_safety
    new_item["image_1_safety"] = image_1_safety
    new_item["alignment_preference"] = alignment_preference
    new_item["quality_preference"] = quality_preference
    new_item["safety_preference"] = safety_preference
    new_item["overall_preference"] = overall_preference
    new_item["label"] = label
    # new_item["pred_gpt4v"] = pred

    # data_list.append(new_item)

    # print("new_item", new_item)
    # input()
    
    save_dir = "validation_gpt4score_new_2_rest_eval.json"

    with open(save_dir, 'a') as f:
        json.dump(new_item, f, indent=4)