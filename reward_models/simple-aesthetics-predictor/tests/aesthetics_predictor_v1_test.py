import pytest
import requests
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1
from aesthetics_predictor.utils import get_model_name_for_v1
from aesthetics_predictor.v1 import convert_from_openai_clip


@pytest.fixture
def sample_image() -> PilImage:
    # the image from https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb
    url = "https://thumbs.dreamstime.com/b/lovely-cat-as-domestic-animal-view-pictures-182393057.jpg"
    return Image.open(requests.get(url, stream=True).raw)


@pytest.mark.parametrize(
    argnames="openai_model_name, expected_img_embeds, expected_prediction",
    argvalues=(
        (
            "openai/clip-vit-base-patch16",
            1.5450,
            4.3533,
        ),
        (
            "openai/clip-vit-base-patch32",
            -0.4287,
            4.4723,
        ),
        (
            "openai/clip-vit-large-patch14",
            0.2653,
            5.0491,
        ),
    ),
)
def test_aesthetics_predictor_v1(
    openai_model_name: str,
    expected_img_embeds: float,
    expected_prediction: float,
    sample_image: PilImage,
) -> None:
    processor = CLIPProcessor.from_pretrained(openai_model_name)
    model = convert_from_openai_clip(openai_model_name)
    model.eval()

    inputs = processor(images=sample_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    actual_img_embeds = outputs.hidden_states.sum().item()
    actual_prediction = outputs.logits.item()

    assert actual_img_embeds == pytest.approx(expected_img_embeds, 0.1)
    assert actual_prediction == pytest.approx(expected_prediction, 0.1)


@pytest.mark.parametrize(
    argnames="openai_model_name, expected_img_embeds, expected_prediction",
    argvalues=(
        (
            "openai/clip-vit-base-patch16",
            1.5450,
            4.3533,
        ),
        (
            "openai/clip-vit-base-patch32",
            -0.4287,
            4.4723,
        ),
        (
            "openai/clip-vit-large-patch14",
            0.2653,
            5.0491,
        ),
    ),
)
def test_load_aesthetics_predictor_v1(
    openai_model_name: str,
    expected_img_embeds: float,
    expected_prediction: float,
    sample_image: PilImage,
):
    model_name = get_model_name_for_v1(openai_model_name)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = AestheticsPredictorV1.from_pretrained(model_name)
    model.eval()

    inputs = processor(images=sample_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    actual_img_embeds = outputs.hidden_states.sum().item()
    actual_prediction = outputs.logits.item()

    assert actual_img_embeds == pytest.approx(expected_img_embeds, 0.1)
    assert actual_prediction == pytest.approx(expected_prediction, 0.1)
