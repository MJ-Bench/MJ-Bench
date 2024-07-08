from typing import Dict, Final, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, logging
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.clip.configuration_clip import CLIPVisionConfig

logging.set_verbosity_error()

URLS: Final[Dict[str, str]] = {
    "openai/clip-vit-base-patch16": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_16_linear.pth",
    "openai/clip-vit-base-patch32": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth",
    "openai/clip-vit-large-patch14": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
}


class AestheticsPredictorV1(CLIPVisionModelWithProjection):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__(config)
        self.predictor = nn.Linear(config.projection_dim, 1)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = outputs[0]  # image_embeds
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

        prediction = self.predictor(image_embeds)

        if not return_dict:
            return (None, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=None,
            logits=prediction,
            hidden_states=image_embeds,
        )


def convert_from_openai_clip(openai_model_name: str) -> AestheticsPredictorV1:
    model = AestheticsPredictorV1.from_pretrained(openai_model_name)
    state_dict = torch.hub.load_state_dict_from_url(URLS[openai_model_name])
    model.predictor.load_state_dict(state_dict)
    model.eval()

    return model
