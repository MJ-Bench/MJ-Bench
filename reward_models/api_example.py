
##### gpt-40/4v example #####
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


client = OpenAI(
# This is the default and can be omitted
api_key="DK3i6fFhq6e0fpmT7kCtSdVDLQ9IKpzk",
base_url="https://azure-openai-api.shenmishajing.workers.dev/v1/")

image_dir = "test.jpg"
image_data = encode_image(image_dir)


while True:
    try:

        response = client.chat.completions.create(
            model="gpt-4o",
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
        
        gpt4_output = response.choices[0].message.content
        print("gpt4_output", gpt4_output)
        break
    except Exception as e:
        if 'out of range' in str(e):
            print(gpt4_output)
        print("Error:", e)
        continue




##### gemini example #####

import google.generativeai as genai
import pathlib

genai.configure(api_key="AIzaSyA5BegcBkbSvu2WNDEtFn7v3OZaINRpV_I")
model = genai.GenerativeModel("gemini-1.5-flash-latest")

prompt = "hi"
image_dir = "test.jpg"

image_data = {
'mime_type': 'image/jpg',
'data': pathlib.Path(image_dir).read_bytes()
}


combined_prompt = [prompt, image_data]

while True:
    try:
        response = model.generate_content(combined_prompt)
        gemini_output = response.text
        print("gemini_output", gemini_output)
        break
    except Exception as e:
        if 'out of range' in str(e):
            print(gemini_output)
        print("Error:", e)
        continue



##### claude example #####
import anthropic
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = anthropic.Anthropic(api_key="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",)

image_dir = "test.jpg"
image_data = encode_image(image_dir)
media_type = "image/jpg"



while True:
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
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
        print("claude_output", claude_output)
        break
    except Exception as e:
        if 'out of range' in str(e):
            print(claude_output)
        print("Error:", e)
        continue

