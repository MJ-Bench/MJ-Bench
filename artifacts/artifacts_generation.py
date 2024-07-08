from utils.image_editor_utils import ImageEditor, load_image
from groundingdino.util.inference import predict
import os
import requests
import io
from PIL import Image
from diffusers import AutoPipelineForText2Image
import torch

def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))



image_generator = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

image_editor = ImageEditor(debugger=False)


if False:
    image_url = 'https://images.pexels.com/photos/4427616/pexels-photo-4427616.jpeg?auto=compress&cs=tinysrgb&w=600'
    local_image_path = 'examples/phd_student_01.png'
    # download_image(image_url, local_image_path)

prompt = f"A high resolution full-body picture of a single person, detailed"
negative_prompt = "skeleton, sketch, deformed face, deformed limbs, bad fingers, implausible"
target_entity = new_entity = "face"
out_dir = "artifacts/examples"
num_generations = 100

count = 0
while count < num_generations:
    local_image_path = f"{out_dir}/human_{count}.jpg"
    if os.path.exists(local_image_path.replace(".jpg", f"_edited_{new_entity}.jpg")):
        count += 1
        continue

    image = image_generator(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, 
        # generator=torch.Generator(device="cuda").manual_seed(0)
    ).images[0]
    image.save(local_image_path)

    # If there is no person in the image, skip
    image_source, image = load_image(local_image_path)
    boxes, logits, phrases = predict(
        model=image_editor.model, 
        image=image, 
        caption="person", 
        box_threshold=image_editor.BOX_TRESHOLD, 
        text_threshold=image_editor.TEXT_TRESHOLD,
    )
    if len(boxes) == 0:
        continue

    try:
        image_inpainting = image_editor.edit_image(local_image_path, target_entity, new_entity, save_dir=out_dir)
    except:
        continue
    count += 1
