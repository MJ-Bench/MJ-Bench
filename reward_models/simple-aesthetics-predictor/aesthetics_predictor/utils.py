import logging

logger = logging.getLogger(__name__)


def get_model_name_for_v1(org_model_name: str) -> str:
    org, model_name = org_model_name.split("/")
    logger.debug(f"org: {org}, model name: {model_name}")

    model_name = "-".join(model_name.split("-")[1:])
    return f"shunk031/aesthetics-predictor-v1-{model_name}"


def get_model_name_for_v2(predictor_head_name: str) -> str:
    predictor_head_name = predictor_head_name.replace("+", "-")
    return f"shunk031/aesthetics-predictor-v2-{predictor_head_name}"
