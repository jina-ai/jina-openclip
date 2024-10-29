import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger

from .coca_model import CoCa
from .loss import (
    ClipLoss,
    CoCaLoss,
    DistillClipLoss,
    SigLipLoss,
    ThreeTowerLoss,
    ThreeTowersCosEmbeddingLoss,
)
from .model import (
    CLIP,
    CustomTextCLIP,
    convert_weights_to_lp,
    get_cast_dtype,
    load_checkpoint,
    set_model_preprocess_cfg,
)
from .openai import load_openai_model
from .pretrained import (
    download_pretrained,
    download_pretrained_from_hf,
    get_pretrained_cfg,
    list_pretrained_tags_by_model,
)
from .tokenizer import DEFAULT_CONTEXT_LENGTH, HFTokenizer, SimpleTokenizer
from .transform import (
    AugmentationCfg,
    PreprocessCfg,
    image_transform_v2,
    merge_preprocess_dict,
    merge_preprocess_kwargs,
)

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / 'model_configs/']
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def _get_hf_config(model_id, cache_dir=None):
    config_path = download_pretrained_from_hf(
        model_id, filename='open_clip_config.json', cache_dir=cache_dir
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_tokenizer(
    model_name: str = '',
    context_length: Optional[int] = None,
    **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX):]
        try:
            config = _get_hf_config(model_name)['model_cfg']
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f'No valid model config found for {model_name}.'

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    require_pretrained: bool = False,
    **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace(
            '/', '-'
        )  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logger.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logger.info(f'Loaded {model_name} model config.')
        else:
            logger.error(
                f'Model config for {model_name} not found; '
                f'available models {list_models()}.'
            )
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg['quick_gelu'] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg['vision_cfg']['patch_dropout'] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg['vision_cfg']['image_size'] = force_image_size

        is_timm_vision_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        is_eva_vision_model = 'eva_model_name' in model_cfg.get('vision_cfg', {})
        is_hf_vision_model = 'hf_vision_model_name' in model_cfg.get('vision_cfg', {})
        is_hf_text_model = 'hf_model_name' in model_cfg.get('text_cfg', {})

        if pretrained_image:
            if is_timm_vision_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            elif is_hf_vision_model:
                model_cfg['vision_cfg']['hf_vision_model_pretrained'] = True
            else:
                assert (
                    False
                ), 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp'
        # or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        if is_hf_text_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = (
                pretrained_hf and not pretrained
            )

        custom_text = (
            model_cfg.pop('custom_text', False) or force_custom_text or is_hf_text_model
        )

        # merge cfg dict w/ kwargs (kwargs overrides cfg)
        model_cfg = dict(model_cfg, **model_kwargs)
        if custom_text:
            if 'multimodal_cfg' in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype, cache_dir=cache_dir)
            else:
                model = CustomTextCLIP(
                    **model_cfg, cast_dtype=cast_dtype, cache_dir=cache_dir
                )
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype, cache_dir=cache_dir)

        is_native_model = not (
            is_timm_vision_model
            or is_hf_vision_model
            or is_hf_text_model
            or is_eva_vision_model
        )
        if precision in ('fp16', 'bf16', 'float16', 'bfloat16'):
            dtype = (
                torch.float16 if precision in ('fp16', 'float16')
                else torch.bfloat16
            )
            # manual mixed precision that matches original OpenAI behaviour
            if is_native_model:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
            else:
                model.to(device=device, dtype=dtype)
                from torch import nn
                from .transformer import LayerNormFp32, LayerNorm

                def _convert_back_to_fp32(m):
                    if isinstance(m, (LayerNorm, LayerNormFp32, nn.LayerNorm)):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                    elif isinstance(m, nn.Parameter) and m.ndim == 0:
                        m.data = m.data.to(torch.float32)
                    elif hasattr(m, 'logit_scale'):
                        m.logit_scale.data = m.logit_scale.data.to(torch.float32)
                    elif hasattr(m, 'mtl_logit_scale'):
                        m.logit_scale.data = m.logit_scale.data.to(torch.float32)
                        m.mtl_logit_scale.data = m.mtl_logit_scale.data.to(
                            torch.float32
                        )

                model.apply(_convert_back_to_fp32)

        elif precision in ('pure_fp16', 'pure_float16'):
            model.to(device=device, dtype=torch.float16)
        elif precision in ('pure_bf16', 'pure_bfloat16'):
            model.to(device=device, dtype=torch.bfloat16)
        else:
            model.to(device=device, dtype=torch.float32)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(
                    pretrained_cfg, cache_dir=cache_dir
                )
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logger.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model '
                    f'{model_name}.Available pretrained tags '
                    f'({list_pretrained_tags_by_model(model_name)}.'
                )
                logger.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True

        elif has_hf_hub_prefix:
            logger.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, '
                f'pretrained: {pretrained}) but not loaded.'
            )

    if output_dict and hasattr(model, 'output_dict'):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size

    set_model_preprocess_cfg(
        model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg)
    )

    return model


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif 'coca' in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif '3towers-text' in args.model.lower():
        return ThreeTowersCosEmbeddingLoss(
            mse_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif '3towers' in args.model.lower():
        return ThreeTowerLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, 'Horovod not currently supported for SigLip'
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    preprocess_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        preprocess_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        preprocess_cfg,
        is_train=False,
    )

    return model, preprocess_cfg, preprocess_train, preprocess_val


def create_model_from_pretrained(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    return_transform: bool = True,
    cache_dir: Optional[str] = None,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
