from utils.image_editor_utils import ImageEditor
from pathlib import Path
import jsonlines
from tqdm import tqdm
import requests
import io
from PIL import Image

def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

query_prompt = """
        Suppose that I have an image that contain two objects. 
        Now I want to remove one of the objects from the image, 
        and replace it with another. Your task is to choose one
        object to place the original one. There are mainly two criteria
        for the new object. 1. It has to be different from the original one,
        and cannot be a synonym of the original one. 
        2. The new object should be as misleading as possible, which means it should
        guide the detection model to think that this new object is the original one,
        however it is not. 
        3. The new object and the other object should be reasonble to co-occur in the same image.
        Now you should provide five candidate objects and generate nothing else.
        For example:
        Original objects: surfboard, person
        Object to replace: surfboard
        New object: skateboard, boat, ship, beach, motorcycle
        Original objects: surfboard, person
        Object to replace: person
        New object: dog, cat, tiger, box, ropes
        Original objects: car, bicycle
        Object to replace: bicycle
        New object: motorcycle, truck, bus, person, charger
        Original objects: {object1}, {object2}
        Object to replace: {object2}
        New object:
        """

image_editor = ImageEditor(debugger=False)


image_url = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/cats.png'
local_image_path = 'examples/cats.png'
# download_image(image_url, local_image_path)


target_entity = "black cat"


# determine the target entity to inpaint
if True:
    new_entity = "dog" 

if False:
    hypernyms_set = []
    hyponyms_set = []
    for ss in wordnet.synsets(concept2):
        for lemma in ss.lemmas():
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    antonyms_set.append(antonym.name())

    for hp in wordnet.synsets(concept2):
        for hypernym in hp.hypernyms():
            hypernyms_set.append(hypernym.name().split(".")[0])
            # then get the hyponym of the hypernym
            for hyponym in hypernym.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponyms_set.append(lemma.name())
            # hypernyms_set.append(lemma.name())
            # antonyms_set.append(lemma.name())
    
    #random choose one from the synonyms
    print(f"hypernyms_set: ", hypernyms_set)
    print(f"hyponyms_set: ", hyponyms_set)

    new_entity = random.choice(hyponyms_set)

if False:
    # using GPT-3.5 to get the new entity
    processed_prompt = query_prompt.format(object1=concept1, object2=concept2)
    print(f"Prompt: ", processed_prompt)
    response = requests.post(
    # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
    'https://api.openai.com/v1/chat/completions',
    headers={'Authorization': f'Bearer {api_key}'},
    json={'model': "gpt-3.5-turbo", "messages": [{"role": "user", "content": processed_prompt}], 'max_tokens': 256, 'n': 1, 'temperature': 0.4}
    )
    data = response.json()
    
    generation = data['choices'][0]['message']['content']
    generation = generation.strip('"')

    entity_list = generation.split(",")

    print(f"entity_list: ", entity_list)
    # randomly choose one
    new_entity = random.choice(entity_list)


image_inpainting = image_editor.edit_image(local_image_path, target_entity, new_entity, save_dir="examples")

# image_inpainting.save(f"examples/{local_image_path}_inpainted_{new_entity}.jpg")
