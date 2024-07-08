import os

import pytest
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import CLIPProcessor

from aesthetics_predictor.v2 import (
    convert_v2_linear_from_openai_clip,
    convert_v2_relu_from_openai_clip,
)


@pytest.fixture
def sample_image() -> PilImage:
    sample_image_path = os.path.join(
        "assets", "a-photo-of-an-astronaut-riding-a-horse.png"
    )
    return Image.open(sample_image_path)


@pytest.fixture
def openai_model_name() -> str:
    return "openai/clip-vit-large-patch14"


@pytest.mark.parametrize(
    argnames="predictor_head_name, expected_img_embeds, expected_prediction",
    argvalues=(
        (
            "sac+logos+ava1-l14-linearMSE",
            1.2222,
            5.7546,
        ),
        (
            "ava+logos-l14-linearMSE",
            1.2222,
            5.6563,
        ),
    ),
)
def test_aesthetics_predictor_v2_linear(
    predictor_head_name: str,
    expected_img_embeds: float,
    expected_prediction: float,
    openai_model_name: str,
    sample_image: PilImage,
) -> None:
    processor = CLIPProcessor.from_pretrained(openai_model_name)
    model = convert_v2_linear_from_openai_clip(predictor_head_name, openai_model_name)
    model.eval()

    inputs = processor(images=sample_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    actual_img_embeds = outputs.hidden_states.sum().item()
    actual_prediction = outputs.logits.item()

    assert actual_img_embeds == pytest.approx(expected_img_embeds, 0.1)
    assert actual_prediction == pytest.approx(expected_prediction, 0.1)


@pytest.mark.parametrize(
    argnames="predictor_head_name, expected_img_embeds, expected_prediction",
    argvalues=(
        (
            "ava+logos-l14-reluMSE",
            1.2222,
            5.3372,
        ),
    ),
)
def test_aesthetics_predictor_v2_relu(
    predictor_head_name: str,
    expected_img_embeds: float,
    expected_prediction: float,
    openai_model_name: str,
    sample_image: PilImage,
) -> None:
    processor = CLIPProcessor.from_pretrained(openai_model_name)
    model = convert_v2_relu_from_openai_clip(predictor_head_name, openai_model_name)
    model.eval()

    inputs = processor(images=sample_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    actual_img_embeds = outputs.hidden_states.sum().item()
    actual_prediction = outputs.logits.item()

    assert actual_img_embeds == pytest.approx(expected_img_embeds, 0.1)
    assert actual_prediction == pytest.approx(expected_prediction, 0.1)
