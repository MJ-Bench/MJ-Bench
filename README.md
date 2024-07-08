# :woman_judge: [**MJ-Bench**: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?]()


<!-- <h3 align="center"><a href="https://arxiv.org/abs/2406.07476" style="color:#9C276A">
**MJ-BENCH**: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation?</a></h3> -->

<h5 align="center"> If our project helps you, please consider giving us a star ⭐. 🥹🙏 </h2>

<h5 align="center">


<div align="center">
  <img src="assets/repo_logo_4.png" width="80%">
</div>

[![hf_space](assets/mj_bench.svg)](https://mj-bench.github.io/)
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed)
[![hf_data](https://img.shields.io/badge/🤗-MSVC-9C276A.svg)](https://huggingface.co/datasets/DAMO-NLP-SG/Multi-Source-Video-Captioning)
[![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476) <br>
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FDAMO-NLP-SG%2FVideoLLaMA2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/DAMO-NLP-SG/VideoLLaMA2?color=critical&label=Issues)](https://github.com/DAMO-NLP-SG/VideoLLaMA2/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/DAMO-NLP-SG/VideoLLaMA2?color=success&label=Issues)](https://github.com/DAMO-NLP-SG/VideoLLaMA2/issues?q=is%3Aissue+is%3Aclosed)  <br>

</h5>

<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>
<!--  may -->

> [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://github.com/DAMO-NLP-SG/Video-LLaMA) <br>
> Hang Zhang, Xin Li, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/Video-LLaMA)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social)](https://github.com/DAMO-NLP-SG/Video-LLaMA) [![arXiv](https://img.shields.io/badge/Arxiv-2306.02858-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2306.02858) <br>

> [**VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding**](https://arxiv.org/abs/2311.16922) <br>
> Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VCD)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VCD.svg?style=social)](https://github.com/DAMO-NLP-SG/VCD)  [![arXiv](https://img.shields.io/badge/Arxiv-2311.16922-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.16922) <br>

</p></details>


## Setup

### Installation
Create environment and install dependencies.
```
conda create -n MM python=3.8
pip install -r requirements.txt
```


### Judge Model Configuration
`config/config.yaml` contains the configuration for the three types of reward models that we will evaluate. You can copy the default configuration to a new file and modify the model_path and api_key to use in your own envionrment.


## Judge Model Evaluation
To get the reward/score from a reward model, simply run
```python
python get_rm_score.py --model [MODEL_NAME] --config_path [CONFIG_PATH] --dataset [DATASET] --local_buffer [LOCAL_BUFFER] --save_dir [SAVE_DIR] --threshold [THRESHOLD]
```

where `MODEL_NAME` is the name of the reward model to evaluate; `CONFIG_PATH` is the path to the configuration file; `DATASET` is the dataset to evaluate on (default is `yuvalkirstain/pickapic_v1`); `LOCAL_BUFFER` specifies a local buffer to cache the images from an online source; `SAVE_DIR` is the directory to save the results; and `THRESHOLD` is the preference threshold for the score-based RMs(i.e. `image_0` is prefered only if `score(image_0) - score(image_1) > THRESHOLD`).


<!-- ## Development Tools

### Additional Installation
For development, you need to install GroundingDINO for image detection and editing. Here is a quick tutorial.

Download pre-trained Swin-T model weights for GroundingDINO.
```
python -m spacy download en_core_web_sm

cd utils/GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### :art: Image Editing Pipeline

`utils/image_editor_utils.py` contains the `ImageEditor` class for the image editing pipeline. To initialize `ImageEditor`, you can specify `inpainting_model_id` for the inpainting model to use. The default inpainting model is `stabilityai/stable-diffusion-2-inpainting`.  You can also set `debugger=True` to save the intermediate source, annotated, and masked images to `utils/image_editor_cache`.


Then you can use the `edit_image` API in `ImageEditor` class for editing. You should provide `local_image_path`, `target_entity` (entity in the image you want to modify), `new_entity` (the entity you want to replace it with). You can also provide the `box_threshold`, `text_threshold` for grounding the target entity or inherit the default values from the `ImageEditor` initialization. You can also specify `save_dir` for the target folder to save the edited image. You can also set `save_dir` to `None` and use the API returned edited image to do further processing or save it by yourself. For example,
```python
from utils.image_editor_utils import ImageEditor

image_editor = ImageEditor(debugger=False)
local_image_path = "path/to/cat.jpg"

# determine the target entity to inpaint
target_entity = "cat"
new_entity = "dog" 

image_inpainting = image_editor.edit_image(local_image_path, target_entity, new_entity, save_dir="path/to/save_dir")
```
A detailed demo is provided in `image_editor_example.py`.


### :t-rex: Evaluation with GroundingDINO

`utils/image_detector_utils.py` contains the `ImageDetector` class for the text-to-image evaluation. To initialize `ImageDetector`, you can specify `args_dict`, `box_threshold`, `text_threshold` which are in-built parameters of GroundingDINO. You can also set `debugger=True` to save cropped images to `cache_dir` in `utils/GroundingDINO/cache`.


Specifially, `ImageDetector` has two API functions for single and batch image detection. The `single_detect` API takes in a single image (str or tensor) and a list of entities and the `batch_detect` API takes in a dict of dict and each dict should contain the following items (box_threshold is optional):
```python
{
    image_0: {"img_path": str, "named_entity": List[str], "box_threshold": float},
    image_1: {"img_path": str, "named_entity": List[str], "box_threshold": float},
    ...
}
```

And the APIs will return the same dictionary with the detection result. Specifically, the result will contain bounding box of the detected entities, the cropped image of the detected entities, and the confidence score of the detection. For example
```python
{
    image_0: {
      "img_path": str,
      "named_entity": List[str],
      "box_threshold": float,
      "detection_result":
        {
          Entity_0: {
              "total_count": int,
              "crop_path": List[str],
              "bbox": List[[x1, y1, x2, y2]],
              "confidence": List[float]
          },
          Entity_1: {
              "total_count": int,
              "crop_path": List[str],
              "bbox": List[[x1, y1, x2, y2]],
              "confidence": List[float]
          },
          ...
        },
      }
      image_1: {
          ...
      },
      ...
}
```
See `image_detect_demo.json` for an example. 

## :wrench: Troubleshooting

#### Error installing `GroundingDINO`

If error `NameError: name '_C' is not defined` is reported, refer to [this issue](https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708) for a quick fix. -->
