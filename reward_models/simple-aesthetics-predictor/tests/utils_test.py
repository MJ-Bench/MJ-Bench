import pytest

from aesthetics_predictor.utils import get_model_name_for_v1, get_model_name_for_v2


@pytest.mark.parametrize(
    argnames="openai_model_name, expected_model_name",
    argvalues=(
        (
            "openai/clip-vit-base-patch16",
            "shunk031/aesthetics-predictor-v1-vit-base-patch16",
        ),
        (
            "openai/clip-vit-base-patch32",
            "shunk031/aesthetics-predictor-v1-vit-base-patch32",
        ),
        (
            "openai/clip-vit-large-patch14",
            "shunk031/aesthetics-predictor-v1-vit-large-patch14",
        ),
    ),
)
def test_get_model_name_for_v1(
    openai_model_name: str, expected_model_name: str
) -> None:
    actual_model_name = get_model_name_for_v1(openai_model_name)
    assert actual_model_name == expected_model_name


@pytest.mark.parametrize(
    argnames="predictor_head_name, expected_model_name",
    argvalues=(
        (
            "sac+logos+ava1-l14-linearMSE",
            "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        ),
        (
            "ava+logos-l14-linearMSE",
            "shunk031/aesthetics-predictor-v2-ava-logos-l14-linearMSE",
        ),
        (
            "ava+logos-l14-reluMSE",
            "shunk031/aesthetics-predictor-v2-ava-logos-l14-reluMSE",
        ),
    ),
)
def test_get_model_name_for_v2(
    predictor_head_name: str, expected_model_name: str
) -> None:
    actual_model_name = get_model_name_for_v2(predictor_head_name)
    assert actual_model_name == expected_model_name
