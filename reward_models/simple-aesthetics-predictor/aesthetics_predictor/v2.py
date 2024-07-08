from collections import OrderedDict
from typing import Dict, Final, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, logging
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.clip.configuration_clip import CLIPVisionConfig

logging.set_verbosity_error()

URLS_LINEAR: Final[Dict[str, str]] = {
    "sac+logos+ava1-l14-linearMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    "ava+logos-l14-linearMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava%2Blogos-l14-linearMSE.pth",
}


URLS_RELU: Final[Dict[str, str]] = {
    "ava+logos-l14-reluMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava%2Blogos-l14-reluMSE.pth",
}


class AestheticsPredictorV2Linear(CLIPVisionModelWithProjection):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__(config)
        self.layers = nn.Sequential(
            nn.Linear(config.projection_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
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

        prediction = self.layers(image_embeds)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct()

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )


class AestheticsPredictorV2ReLU(AestheticsPredictorV2Linear):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.layers = nn.Sequential(
            nn.Linear(config.projection_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.post_init()


def convert_v2_linear_from_openai_clip(
    predictor_head_name: str,
    openai_model_name: str = "openai/clip-vit-large-patch14",
) -> AestheticsPredictorV2Linear:
    model = AestheticsPredictorV2Linear.from_pretrained(openai_model_name)

    state_dict = torch.hub.load_state_dict_from_url(
        URLS_LINEAR[predictor_head_name], map_location="cpu"
    )
    assert isinstance(state_dict, OrderedDict)

    # remove `layers.` from the key of the state_dict
    state_dict = OrderedDict(
        ((k.replace("layers.", ""), v) for k, v in state_dict.items())
    )
    model.layers.load_state_dict(state_dict)

    model.eval()

    return model


def convert_v2_relu_from_openai_clip(
    predictor_head_name: str,
    openai_model_name: str = "openai/clip-vit-large-patch14",
) -> AestheticsPredictorV2ReLU:
    model = AestheticsPredictorV2ReLU.from_pretrained(openai_model_name)

    state_dict = torch.hub.load_state_dict_from_url(
        URLS_RELU[predictor_head_name], map_location="cpu"
    )
    assert isinstance(state_dict, OrderedDict)

    # remove `layers.` from the key of the state_dict
    state_dict = OrderedDict(
        ((k.replace("layers.", ""), v) for k, v in state_dict.items())
    )
    model.layers.load_state_dict(state_dict)

    model.eval()

    return model
