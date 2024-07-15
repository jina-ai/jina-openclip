"""CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License,
Copyright (c) 2021 OpenAI.
"""

import copy
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from .eva_model import EVAVisionTransformer
from .hf_model import HFTextEncoder, HFVisionEncoder
from .modified_resnet import ModifiedResNet
from .pretrained import download_pretrained, get_pretrained_cfg
from .timm_model import TimmModel
from .transformer import (
    Attention,
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    TextTransformer,
    VisionTransformer,
    text_global_pool,
)
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    proj_type: Optional[str] = None
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # Pretrained model checkpoint
    # Load the vision tower from this checkpoint
    pt_model_name: Optional[str] = None
    pt_proj_exclude: bool = True

    # TIMM specific vision tower config
    # a valid model name overrides layers, width, patch_size
    timm_model_name: Optional[str] = None
    # use (imagenet) pretrained weights for named model
    timm_model_pretrained: bool = False
    # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_pool: str = 'avg'
    # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj: str = 'linear'
    # enable bias final projection
    timm_proj_bias: bool = False
    # head dropout
    timm_drop: float = 0.0
    #attention dropout
    timm_attn_drop: float = 0.0
    #mpl hidden layer dropout, is called proj_drop for timm
    timm_proj_drop: float = 0.0
    # backbone stochastic depth
    timm_drop_path: Optional[float] = None

    # hf specific vision tower config
    # a valid model name overrides layers, width, patch_size
    hf_vision_model_name: Optional[str] = None
    # use (imagenet) pretrained weights for named model
    hf_vision_model_pretrained: bool = False
    # feature pooling for timm model ('tok', 'avg', '')
    hf_vision_pool: str = 'avg'
    # linear projection for hf vision encoder output ('linear', 'mlp', '')
    hf_vision_proj: str = 'linear'
    # enable bias final projection
    hf_vision_proj_bias: bool = False
    # vision model hidden states dropout
    hf_vision_hidden_states_drop: float = 0.0
    # vision model attention probabilities dropout
    hf_vision_attn_drop: float = 0.0
    # backbone stochastic depth
    hf_vision_drop_path: Optional[float] = None

    eva_model_name: str = (
        None  # a valid eva model name overrides layers, width, patch_size
    )
    eva_pretrained: str = None
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16  # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False
    drop_path_rate: Optional[float] = None  # drop path rate


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_type: Optional[str] = None
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # Pretrained model checkpoint
    # Load the text tower from this checkpoint
    pt_model_name: Optional[str] = None
    pt_proj_exclude: bool = True

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models
    hf_trust_remote_code: bool = False
    hf_model_revision: Optional[str] = None
    hf_model_code_revision: Optional[str] = None


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision in {'bf16', 'bfloat16'}:
        cast_dtype = torch.bfloat16
    elif precision in {'fp16', 'float16'}:
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _load_state_dict(checkpoint_path: str, map_location: str = 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ['input_resolution', 'context_length', 'vocab_size']:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint

    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    root: Optional[str] = None,
    exclude: Optional[List[str]] = None,
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        from .big_vision import load_big_vision_weights

        model: CustomTextCLIP

        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = _load_state_dict(checkpoint_path)

    root = root or ''
    root_modules = root.split('.')
    root_modules = [rm for rm in root_modules if rm != '']
    num_root_modules = len(root_modules)
    exclude = exclude or []

    if num_root_modules > 0:
        _state_dict = {}
        for k, v in state_dict.items():
            _k_modules = k.split('.')
            _root_k_modules = _k_modules[:num_root_modules]
            if _root_k_modules == root_modules:
                _new_k = '.'.join(_k_modules[num_root_modules:])
                _state_dict[_new_k] = v
        if len(_state_dict) == 0:
            raise ValueError(
                f"Got an empty state dict after filtering using root '{root}'"
            )
        state_dict = copy.deepcopy(_state_dict)

    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(
        model, 'positional_embedding'
    ):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # If loading a non-SigLIP model for SigLIP training. See
    # https://github.com/mlfoundations/open_clip/issues/712
    # if "logit_bias" not in state_dict and model.logit_bias is not None:
    #     state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])
    # Certain text transformers no longer expect position_ids after transformers==4.31

    # remove logit biases and scale for 3-towers
    # if 'logit_bias' in state_dict:
    #     del state_dict['logit_bias']
    # if 'logit_scale' in state_dict:
    #     del state_dict['logit_scale']

    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    if exclude:
        _state_dict = {}
        for k, v in state_dict.items():
            if any([k.startswith(e) for e in exclude]):
                continue
            _state_dict[k] = v

        state_dict = copy.deepcopy(_state_dict)

        if len(_state_dict) == 0:
            raise ValueError(
                f"Got an empty state dict after filtering using exclude '{exclude}'"
            )

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)

    return incompatible_keys


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: Union[Dict, CLIPVisionCfg],
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and
    # more memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout
            if vision_cfg.patch_dropout > 0
            else None,
            attn_drop=vision_cfg.timm_attn_drop,
            proj_drop=vision_cfg.timm_proj_drop,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif vision_cfg.hf_vision_model_name:
        visual = HFVisionEncoder(
            vision_cfg.hf_vision_model_name,
            output_dim=embed_dim,
            proj_type=vision_cfg.hf_vision_proj,
            proj_bias=vision_cfg.hf_vision_proj_bias,
            pool_type=vision_cfg.hf_vision_pool,
            pretrained=vision_cfg.hf_vision_model_pretrained,
            output_tokens=vision_cfg.output_tokens,
            image_size=vision_cfg.image_size,
            attn_drop=vision_cfg.hf_vision_attn_drop,
            hidden_drop=vision_cfg.hf_vision_hidden_states_drop,
            drop_path=vision_cfg.hf_vision_drop_path,
        )
    elif vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width

        if vision_cfg.fusedLN:
            try:
                from apex.normalization import FusedLayerNorm
                norm_layer = FusedLayerNorm
            except ModuleNotFoundError or ImportError:
                norm_layer = (
                    LayerNormFp32
                    if cast_dtype in (torch.float16, torch.bfloat16)
                    else LayerNorm
                )
                print("Please 'pip install apex'")
        else:
            norm_layer = (
                LayerNormFp32
                if cast_dtype in (torch.float16, torch.bfloat16)
                else LayerNorm
            )

        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=False,  # False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(norm_layer, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
            proj_type=vision_cfg.proj_type,
        )
        _model_name = vision_cfg.eva_model_name
        if _model_name is not None:
            _tag = vision_cfg.eva_pretrained.replace(
                '/', '-'
            )  # for callers using old naming with / in ViT names
            pretrained_image_cfg = get_pretrained_cfg(model=_model_name, tag=_tag)
            if pretrained_image_cfg:
                visual_checkpoint_path = download_pretrained(
                    pretrained_image_cfg, cache_dir=cache_dir
                )
                state_dict = _load_state_dict(visual_checkpoint_path)
                state_dict = {
                    key: value for key, value in state_dict.items() if 'rope' not in key
                }
                visual.load_state_dict(state_dict, strict=False)
            else:
                _error_str = (
                    f"No checkpoint for model '{_model_name}' found neither locally "
                    f'nor remotely'
                )
                logger.exception(_error_str)
                raise RuntimeError(_error_str)

    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            proj_type=vision_cfg.proj_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        ckpt = vision_cfg.pt_model_name
        if ckpt is not None:
            logger.info(f'Downloading pretrained model {ckpt} ...')
            _model, _tag = ckpt.split(' ')
            _ckpt_path = download_pretrained(
                get_pretrained_cfg(model=_model, tag=_tag), cache_dir=cache_dir
            )
            if _ckpt_path:
                exclude = []
                if visual.proj is None or vision_cfg.pt_proj_exclude:
                    exclude.append('proj')

                logger.info(f'Loading pretrained model from {_ckpt_path} ...')
                load_checkpoint(
                    visual, _ckpt_path, strict=False, root='visual', exclude=exclude
                )
            else:
                _error_str = (
                    f"No checkpoint for model '{ckpt}' found neither locally nor "
                    f'remotely'
                )
                logger.exception(_error_str)
                raise RuntimeError(_error_str)

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: Union[Dict, CLIPTextCfg],
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
            trust_remote_code=text_cfg.hf_trust_remote_code,
            revision=text_cfg.hf_model_revision,
            code_revision=text_cfg.hf_model_code_revision,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        ckpt = text_cfg.pt_model_name
        if ckpt is not None:
            logger.info(f'Downloading pretrained model {ckpt} ...')
            _model, _tag = ckpt.split(' ')
            _ckpt_path = download_pretrained(
                get_pretrained_cfg(model=_model, tag=_tag), cache_dir=cache_dir
            )
            if _ckpt_path:
                exclude = []
                if text.text_projection is None or text_cfg.pt_proj_exclude:
                    exclude.append('text_projection')

                logger.info(f'Loading pretrained model from {_ckpt_path} ...')
                load_checkpoint(
                    text, _ckpt_path, strict=False, root='text', exclude=exclude
                )
            else:
                _error_str = (
                    f'No checkpoint for model {ckpt} found neither locally nor '
                    f'remotely'
                )
                logger.exception(_error_str)
                raise RuntimeError(_error_str)
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        freeze_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype, cache_dir
        )
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, cache_dir)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * init_logit_scale, requires_grad=not freeze_logit_scale
        )
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.ones([]) * init_logit_bias, requires_grad=not freeze_logit_scale
            )
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

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


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        freeze_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype, cache_dir
        )
        self.text = _build_text_tower(
            embed_dim, text_cfg, quick_gelu, cast_dtype, cache_dir
        )
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(
            torch.ones([]) * init_logit_scale, requires_grad=not freeze_logit_scale
        )
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.ones([]) * init_logit_bias, requires_grad=not freeze_logit_scale
            )
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

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


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [
                *[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                'in_proj_bias',
                'bias_k',
                'bias_v',
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, 'text_projection', None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, 'proj', None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(
                k.startswith(p)
                for p in (
                    'text_projection',
                    'positional_embedding',
                    'token_embedding',
                    'transformer',
                    'ln_final',
                )
            ):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
):
    vit = 'visual.proj' in state_dict

    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
            ]
        )
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: List = [
            len(
                set(
                    k.split('.')[2]
                    for k in state_dict
                    if k.startswith(f'visual.layer{b}')
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round(
            (state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict['visual.attnpool.positional_embedding'].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks')
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ['input_resolution', 'context_length', 'vocab_size']:
        state_dict.pop(key, None)
    convert_weights_to_fp16(
        model
    )  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,),
        ),
    )
    model.visual.image_size = image_size
    return model


def resize_pos_embed(
    state_dict, model, interpolation: str = 'bicubic', antialias: bool = True
):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = (
        1  # FIXME detect different token configs (ie no class token, or more)
    )
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = (
            old_pos_embed[:extra_tokens],
            old_pos_embed[extra_tokens:],
        )
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logger.info(
        'Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size
    )
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1
    ).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1
    )[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(
    state_dict, model, interpolation: str = 'linear', antialias: bool = False
):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logger.info(
        'Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos
    )
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg[
        'mean'
    ]  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(
        preprocess_cfg
    )  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg
