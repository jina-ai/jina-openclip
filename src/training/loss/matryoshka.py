from collections import defaultdict
from typing import Optional, Sequence

import torch
from torch import nn


class MatryoshkaOperator(nn.Module):
    def __init__(
        self,
        loss: nn.Module,
        dims: Sequence[int] = (16, 32, 64, 128, 256, 512),
        weights: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self._loss = loss
        if weights:
            assert len(weights) == len(dims)
        self._dims = dims
        self._weights = weights if weights else [1] * len(self._dims)

    @staticmethod
    def _is_feature_tensor(obj):
        return (
            torch.is_tensor(obj)
            and torch.is_floating_point(obj)
            and len(obj.shape) == 2
        )

    def forward(self, *args, **kwargs):
        _tensor_args_idxs = [
            i for i, arg in enumerate(args) if self._is_feature_tensor(arg)
        ]
        _tensor_kwargs_idxs = [
            k for k, v in kwargs.items() if self._is_feature_tensor(v)
        ]
        losses = defaultdict(float)
        output_dict = kwargs.pop('output_dict', False)

        for dim, weight in zip(self._dims, self._weights):
            _args = [
                arg if i not in _tensor_args_idxs else arg[..., :dim].contiguous()
                for i, arg in enumerate(args)
            ]
            _kwargs = {
                k: v if k not in _tensor_kwargs_idxs else v[..., :dim].contiguous()
                for k, v in kwargs.items()
            }
            _composite_loss = self._loss(*_args, **_kwargs, output_dict=output_dict)
            if output_dict:
                for k, v in _composite_loss.items():
                    losses[k] += v * weight
            else:
                losses['loss'] += _composite_loss * weight

        return losses if output_dict else losses['loss']
