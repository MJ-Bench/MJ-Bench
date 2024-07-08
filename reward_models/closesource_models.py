import os
from openai import OpenAI
import anthropic
import google.generativeai as genai
from PIL import Image
from IPython.display import Markdown
import time
import base64
import requests
import pathlib
import re
# import yaml



metric_narratives = {
    5: ["Extremely Poor", "Poor", "Average", "Good", "Outstanding"],
    7: ["Extremely Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Outstanding"],
    10: ["Extremely Poor", "Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good", "Excellent", "Outstanding"]
}



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_vlm_output(text, patterns, aid_text="assistant"):
    text = text.split(aid_text)[-1]
    results = dict()
    for k in patterns.keys():
        pattern = patterns[k]
        result_match = re.findall(pattern, text)
        if len(result_match) != 0:
            results[k] = result_match[-1]
        else:
            results[k] = "N/A"
    return text, results

def parse_scores(scores, multi_image, perspective, metric_type, metric_scale):

    metric_pattern0 ="|".join(metric_narratives[metric_scale])
    metric_pattern1 = "|".join([f"\[{metric}\]" for metric in metric_narratives[metric_scale]])
    metric_pattern2 = "|".join([f"\[{metric.lower()}\]" for metric in metric_narratives[metric_scale]])
    metric_pattern3 = "|".join([f"\[{metric.upper()}\]" for metric in metric_narratives[metric_scale]])

    if multi_image:
        parse_patterns = {
            "IMAGE-1 RATING": re.compile(r'IMAGE-1 RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3)),
            "IMAGE-2 RATING": re.compile(r'IMAGE-2 RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3)),
            "BETTER IMAGE": re.compile(r'BETTER IMAGE: ([0-2]|IMAGE-1|IMAGE-2|IMAGE 1|IMAGE 2|image-1|image-2|image 1|image 2)'),
        }
    else:
        parse_patterns = {
            "RATING": re.compile(r'{4}-RATING: (\d+\.?\d?|\d|{0}|{1}|{2}|{3})'.format(metric_pattern0, metric_pattern1, metric_pattern2, metric_pattern3, perspective.replace("_", "-").upper())),
            "ANASYLIS": re.compile(r'ANASYLIS: (.+)'),
        }

    if multi_image:
        vlm_output, parsed_results = parse_vlm_output(scores, parse_patterns)
        score_0 = parsed_results["IMAGE-1 RATING"]
        score_1 = parsed_results["IMAGE-2 RATING"]
        better_id = parsed_results["BETTER IMAGE"]
        if better_id.upper() == "IMAGE-1" or better_id.upper() == "IMAGE 1":
            better_id = 1
        elif better_id.upper() == "IMAGE-2" or better_id.upper() == "IMAGE 2":
            better_id = 2
    else:
        output_0, parsed_results_0 = parse_vlm_output(scores[0], parse_patterns)
        score_0 = parsed_results_0["RATING"]
        output_1, parsed_results_1 = parse_vlm_output(scores[1], parse_patterns)
        
        score_1 = parsed_results_1["RATING"]   

    return score_0, score_1

class Scorer:
    def __init__(self, model_name, model_path, api_key, base_url):
        self.model_name = model_name
        self.model_path = model_path
        self.api_key = api_key
        self.base_url = base_url

        if "gpt" in model_name:
            self.client = OpenAI(
            # This is the default and can be omitted
            api_key=self.api_key,
            base_url=self.base_url
        )

            self.get_score = self.gpt_score
            
        elif "gemini" in model_name:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_path)
            self.get_score = self.gemini_score
            
        elif "claude" in model_name:
            self.client = anthropic.Anthropic(
            api_key=self.api_key,
            )
            self.get_score = self.claude_score

        else:
            raise ValueError(f"Model {model_name} not found")
        

    def gpt_score(self, images, prompt):

        if len(images) == 1:
            image_dir = images[0]
            image_data = encode_image(image_dir)

            while True:
                try:
                    # print("prompt", prompt)
                    # print("self.model_path", self.model_path)
                    # print("self.api_key", self.api_key)
                    # print("self.base_url", self.base_url)

                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{prompt}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    }
                                ],
                            }
                        ],
                        max_tokens=256,
                        temperature=1
                    )
                    
                    # input()
                    # data = response.json()
                    # print("data", data)
                    gpt4_output = response.choices[0].message.content
                    # print("gpt4_output", gpt4_output)
                    analysis = gpt4_output.split("ANALYSIS:")[1].split("RATING:")[0]
                    rating = gpt4_output.split("RATING:")[1]

                    break
                except Exception as e:
                    if "ResponsibleAIPolicyViolation" in str(e):
                        analysis = str(e)
                        rating = 5
                        break
                    print("Error:", e)
                    time.sleep(2)
                
            return rating, analysis
        
        if len(images) > 1:

            image_0_dir, image_1_dir = images
            image_data_0 = encode_image(image_0_dir)
            image_data_1 = encode_image(image_1_dir)

            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{prompt}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data_0}"
                                        },
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data_1}"
                                        },
                                    }
                                ],
                            }
                        ],
                        max_tokens=256,
                        temperature=1
                    )
                    
                    gpt4_output = response.choices[0].message.content

                    image_1_rating = gpt4_output.split("IMAGE-1 RATING:")[1].split("IMAGE-2 RATING:")[0]
                    image_2_rating = gpt4_output.split("IMAGE-2 RATING:")[1].split("BETTER IMAGE:")[0]
                    preference = gpt4_output.split("BETTER IMAGE:")[1].split("ANALYSIS OF CHOICE:")[0]
                    analysis = gpt4_output.split("ANALYSIS OF CHOICE:")[1]

                    break

                except Exception as e:
                    if "ResponsibleAIPolicyViolation" in str(e):
                        image_1_rating = 5
                        image_2_rating = 5
                        preference = -1
                        analysis = str(e)
                        break
                    print("Error:", e)
                    time.sleep(2)


            return image_1_rating, image_2_rating, preference, analysis



    def claude_score(self, images, prompt):
        
        media_type = "image/jpeg"
        
        if len(images) == 1:
            image_dir = images[0]
            image_data = encode_image(image_dir)
            

            while True:
                try:
                    message = self.client.messages.create(
                        model=self.model_path,
                        max_tokens=256,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ],
                            }
                        ],
                    )
                    claude_output = message.content[0].text

                    analysis = claude_output.split("ANALYSIS:")[1].split("RATING:")[0]
                    rating = claude_output.split("RATING:")[1]

                    break
            
                except Exception as e:
                    print("Error:", e)
                    if 'out of range' in str(e):
                        rating = 5
                        analysis = claude_output
                        # return rating
                        print(claude_output)
                        
                    time.sleep(2)



            return rating, analysis

        if len(images) > 1:
                
                image_0_dir, image_1_dir = images
                image_data_0 = encode_image(image_0_dir)
                image_data_1 = encode_image(image_1_dir)
    

                while True:
                    try:
                        message = self.client.messages.create(
                            model=self.model_path,
                            max_tokens=256,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": image_data_0,
                                            },
                                        },
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": image_data_1,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": prompt
                                        }
                                    ],
                                }
                            ],
                        )
                        claude_output = message.content[0].text
                        

                        image_1_rating = claude_output.split("IMAGE-1 RATING:")[1].split("IMAGE-2 RATING:")[0]
                        image_2_rating = claude_output.split("IMAGE-2 RATING:")[1].split("BETTER IMAGE:")[0]
                        preference = claude_output.split("BETTER IMAGE:")[1].split("ANALYSIS OF CHOICE:")[0]
                        analysis = claude_output.split("ANALYSIS OF CHOICE:")[1]

                        break
                
                    except Exception as e:
                        print("Error:", e)
                        time.sleep(2)
    
                return image_1_rating, image_2_rating, preference, analysis

    
    def gemini_score(self, images, prompt):

        if len(images) == 1:
            image_dir = images[0]
            image_data = {
            'mime_type': 'image/jpg',
            'data': pathlib.Path(image_dir).read_bytes()
            }
            

            combined_prompt = [prompt, image_data]

            while True:
                try:
                    response = self.model.generate_content(combined_prompt)
                    gemini_output = response.text
                    analysis = gemini_output.split("ANALYSIS:")[1].split("RATING:")[0]
                    rating = gemini_output.split("RATING:")[1]
                    break
                except Exception as e:
                    if 'out of range' in str(e):
                        rating = 5
                        analysis = gemini_output
                        # return rating
                        print(gemini_output)
                    print("Error:", e)
                    time.sleep(2)

            return rating, analysis

        if len(images) > 1:
                
                image_0_dir, image_1_dir = images
                image_data_0 = {
                'mime_type': 'image/jpg',
                'data': pathlib.Path(image_0_dir).read_bytes()
                }
                image_data_1 = {
                'mime_type': 'image/jpg',
                'data': pathlib.Path(image_1_dir).read_bytes()
                }
    
                combined_prompt = [prompt, image_data_0, image_data_1]

                while True:
                    try:
                        response = self.model.generate_content(combined_prompt)
                        gemini_output = response.text

                        image_1_rating = gemini_output.split("IMAGE-1 RATING:")[1].split("IMAGE-2 RATING:")[0]
                        image_2_rating = gemini_output.split("IMAGE-2 RATING:")[1].split("BETTER IMAGE:")[0]
                        preference = gemini_output.split("BETTER IMAGE:")[1].split("ANALYSIS OF CHOICE:")[0]
                        analysis = gemini_output.split("ANALYSIS OF CHOICE:")[1]
                        break
                    except Exception as e:
                        print("Error:", e)
                        time.sleep(2)
    
                return image_1_rating, image_2_rating, preference, analysis
      