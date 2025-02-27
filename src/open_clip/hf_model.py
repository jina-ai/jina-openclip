import re
from typing import Any, Dict, Optional, Tuple
import numpy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

"""
HF architecture mapping
"""

_HF_ARCH_DICT = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    'roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    'xlm-roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    'mt5': {
        'config_names': {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            'context_length': '',
            'vocab_size': 'vocab_size',
            'width': 'd_model',
            'heads': 'num_heads',
            'layers': 'num_layers',
            'layer_attr': 'block',
            'token_embeddings_attr': 'embed_tokens',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    'bert': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
        },
        'pooler': 'cls_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/m2m_100
    'm2m_100': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'd_model',
            'heads': 'encoder_attention_heads',
            'layers': 'encoder_layers',
        },
        'pooler': 'cls_pooler',
    },
}


"""
Pooling functions
"""

_POOLERS = {}


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """
    Max pooling
    """

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """
    CLS token pooling
    """

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, _: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


"""
HF text model
"""


class HFTextEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj_type: str = None,
        proj_bias: bool = False,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        model_config_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        model_config_kwargs = model_config_kwargs or {}

        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
            )
            self.config.update(model_config_kwargs)
            create_func, model_args = (
                (AutoModel.from_pretrained, model_name_or_path)
                # if pretrained
                # else (AutoModel.from_config, self.config)
            )
            # TODO: do all model configs have this attribute?
            #  PretrainedConfig does so yes??
            if (
                hasattr(self.config, 'is_encoder_decoder')
                and self.config.is_encoder_decoder
            ):
                self.transformer = create_func(
                    model_args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    **model_config_kwargs
                )
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(
                    model_args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    add_pooling_layer=False,
                    code_revision=code_revision,
                    **model_config_kwargs
                )
        else:
            self.config = config
            self.config.update(model_config_kwargs)
            self.transformer = AutoModel.from_config(
                self.config,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
            )

        # FIXME downstream users of OpenCLIP models use these attr,
        #  need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        pooler_type = pooler_type or _HF_ARCH_DICT[self.config.model_type]['pooler']
        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(
            self.config, _HF_ARCH_DICT[self.config.model_type]['config_names']['width']
        )
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif (d_model != output_dim) or proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )
        self._lora_adapter_ids = None
        lora_adapters_to_train = self.config.lora_adaptations
        if lora_adapters_to_train:
            lora_tasks = {
                task: num
                for num, task in enumerate(self.transformer.config.lora_adaptations)
            }
            for lora_adapter in lora_adapters_to_train:
                if lora_adapter not in lora_tasks:
                    raise ValueError(
                        f"Unsupported lora adapter '{lora_adapter}'. "
                        f"Supported adapters are: "
                        f"{', '.join(self.transformer.config.lora_adaptations)}."
                    )

            self._lora_adapter_ids = [
                lora_tasks[name] for name in lora_adapters_to_train
            ]
            if len(self._lora_adapter_ids) > 2:
                raise ValueError(
                    'Tuning more than two LoRA adapters is not supported. '
                    'Please reduce the number.'
                )

    def _create_adapter_mask(
        self, num_examples: int, row_sizes: Optional[numpy.ndarray] = None
    ):
        if len(self._lora_adapter_ids) == 1:
            adapter_mask = torch.full(
                (num_examples,),
                self._lora_adapter_ids[0],
                dtype=torch.int32,
            )
        elif len(self._lora_adapter_ids) == 2:
            if row_sizes is None:
                raise ValueError(
                    'When tuning 2 LoRA adapters `row_sizes` must not be `None`.'
                )
            row_sizes = torch.tensor(row_sizes)
            cumulative_sizes = torch.cumsum(row_sizes, dim=0)
            first_indices = torch.cat((torch.tensor([0]), cumulative_sizes[:-1]))
            remaining_indices = torch.cat(
                [
                    torch.arange(first_indices[i] + 1, first_indices[i] + row_sizes[i])
                    for i in range(len(row_sizes))
                ]
            )
            adapter_mask = torch.empty(
                num_examples, dtype=torch.int32, device=self._device
            )
            adapter_mask[first_indices] = self._lora_adapter_ids[0]
            adapter_mask[remaining_indices] = self._lora_adapter_ids[1]
        else:
            raise ValueError(
                'Tuning more than two LoRA adapters is not supported. '
                'Please reduce the number.'
            )
        return adapter_mask

    def forward(self, x: torch.Tensor):
        attn_mask = (x != self.config.pad_token_id).long()
        num_examples = x.shape[0]
        lora_args = {}
        if self._lora_adapter_ids:
            lora_args['adapter_mask'] = self._create_adapter_mask(
                        num_examples
                    ).to(x.device)
        out = self.transformer(input_ids=x, attention_mask=attn_mask, **lora_args)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seq_len) != self.pooler.cls_token_position, :
            ]
            if isinstance(self.pooler, ClsPooler)
            else out.last_hidden_state
        )

        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )
            return

        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, _=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass


"""
HF vision model
"""


class HFVisionEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        image_size: int,
        output_dim: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pool_type: str = 'tok',
        proj_type: Optional[str] = None,
        proj_bias: bool = False,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
        is_composite: bool = False,
        is_causal_lm: bool = False,
        vision_config_field: Optional[str] = None,
        vision_model_field: Optional[str] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.image_size = (image_size, image_size)

        if is_composite:
            assert vision_config_field
            assert vision_model_field

        if is_composite:
            _model_kwargs = {}
        else:
            _model_kwargs = model_kwargs or {}

        _model_class = AutoModelForCausalLM if is_causal_lm else AutoModel

        if pretrained:
            transformer = _model_class.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                **_model_kwargs,
            )
            config = transformer.config
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                **_model_kwargs,
            )
            transformer = _model_class.from_config(config)

        self.config = (
            getattr(config, vision_config_field)
            if is_composite else config
        )
        self.transformer = (
            getattr(transformer, vision_model_field)
            if is_composite else transformer
        )

        if 'dinov2' in model_name_or_path:
            self.transformer.embeddings.mask_token.requires_grad = False

        self._transformer_forward = self._transformer_forward_generic
        self._set_grad_checkpointing = self._set_grad_checkpointing_generic
        if 'Florence-2' in model_name_or_path:
            self._transformer_forward = self._transformer_forward_florence2
            self._set_grad_checkpointing = self._set_grad_checkpointing_florence2
        elif 'InternViT' in model_name_or_path:
            self._set_grad_checkpointing = self._set_grad_checkpointing_internvit

        assert pool_type in ('tok', 'avg', 'none')
        self.pool_type = pool_type

        d_model = (
            self.config.hidden_size if hasattr(self.config, 'hidden_size')
            else self.config.dim_embed[-1]
        )
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )

    def _transformer_forward_generic(self, x: torch.Tensor):
        return self.transformer(x)[0]

    def _transformer_forward_florence2(self, x: torch.Tensor):
        return self.transformer.forward_features_unpool(x)

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        # returns a tuple of (final hidden states, token pooled outputs)
        x = self._transformer_forward(x)
        pooled, tokens = self._global_pool(x)
        projected = self.proj(pooled)

        return projected

    def lock(self, unlocked_layers: int = 0, freeze_bn_stats: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_bn_stats) if 'LayerNorm' in n.split('.') else False
                )
            return

        # TODO: make it work if unlocked_layers !=0
        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_bn_stats) if 'LayerNorm' in n.split('.') else False
                )

    def _set_grad_checkpointing_generic(self):
        self.transformer.gradient_checkpointing_enable()

    def _set_grad_checkpointing_florence2(self):
        self.transformer.enable_checkpoint = True

    def _set_grad_checkpointing_internvit(self):
        self.transformer.encoder.gradient_checkpointing = True

    @torch.jit.ignore
    def set_grad_checkpointing(self, *_, **__):
        self._set_grad_checkpointing()

    def init_parameters(self):
        pass
