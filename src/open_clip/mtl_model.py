from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from .hf_model import HFTextEncoder
from .model import (
    CLIPTextCfg,
    CLIPVisionCfg,
    CustomTextCLIP,
    _build_text_tower,
    _build_vision_tower,
)
from .transformer import TextTransformer, VisionTransformer


class MTLPairCLIP(CustomTextCLIP):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        teacher_cfg: Union[CLIPVisionCfg, CLIPTextCfg],
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        cache_dir: Optional[str] = None,
        tie_projections: bool = False,
    ):
        super(MTLPairCLIP, self).__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            cast_dtype=cast_dtype,
            output_dict=output_dict,
            cache_dir=cache_dir,
        )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {
                'image_features': image_features,
                'text_features': text_features,
                'logit_scale': self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        return image_features, text_features, self.logit_scale.exp()
