from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as f


def _neighbour_exchange(
    from_rank: int, to_rank: int, tensor: torch.Tensor, group: Any = None
):
    """
    Sends `tensor` to device with rank `to_rank`. Receives a tensor of the same shape
    as `tensor` from device with rank `from_rank`. Ranks are relative to group, if
    `group` is provided. Returns received tensor.
    """
    _tensor_recv = torch.zeros_like(tensor)

    _send_operation = torch.distributed.P2POp(
        torch.distributed.isend, tensor, to_rank, group=group,
    )
    _recv_operation = torch.distributed.P2POp(
        torch.distributed.irecv, _tensor_recv, from_rank, group=group,
    )

    operations = torch.distributed.batch_isend_irecv([_send_operation, _recv_operation])
    for operation in operations:
        operation.wait()

    return _tensor_recv


def _neighbour_exchange_bidirectional(
    left_rank: int,
    right_rank: int,
    tensor_to_left: torch.Tensor,
    tensor_to_right: torch.Tensor,
    group: Any = None,
):
    """
    Sends `tensor_to_left` to device with rank `left_rank`. Sends `tensor_to_right` to
    device with rank `right_rank`. Receives a tensor of the same shape as
    `tensor_to_right` from device with rank `left_rank`. Receives a tensor of the same
    shape as  `tensor_to_left` from device with rank `right_rank`. Ranks are relative
    to group, if `group` is provided. Returns received tensors.
    """
    _tensor_from_left = torch.zeros_like(tensor_to_right)
    _tensor_from_right = torch.zeros_like(tensor_to_left)

    _send_operation_left = torch.distributed.P2POp(
        torch.distributed.isend, tensor_to_left, left_rank, group=group,
    )
    _send_operation_right = torch.distributed.P2POp(
        torch.distributed.isend, tensor_to_right, right_rank, group=group,
    )
    _recv_operation_left = torch.distributed.P2POp(
        torch.distributed.irecv, _tensor_from_left, left_rank, group=group,
    )
    _recv_operation_right = torch.distributed.P2POp(
        torch.distributed.irecv, _tensor_from_right, right_rank, group=group,
    )

    operations = torch.distributed.batch_isend_irecv(
        [
            _send_operation_right,
            _send_operation_left,
            _recv_operation_right,
            _recv_operation_left,
        ]
    )
    for operation in operations:
        operation.wait()

    return _tensor_from_right, _tensor_from_left


# noinspection PyMethodOverriding
class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, from_rank: int, to_rank: int, group: Any, tensor: torch.Tensor
    ):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return _neighbour_exchange(
            from_rank=from_rank, to_rank=to_rank, tensor=tensor, group=group
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Any):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(
    from_rank: int, to_rank: int, tensor: torch.Tensor, group: Any = None
):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


# noinspection PyMethodOverriding
class NeighbourExchangeBidirectional(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        left_rank: int,
        right_rank: int,
        group: Any,
        tensor_to_left: torch.Tensor,
        tensor_to_right: torch.Tensor,
    ):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return _neighbour_exchange_bidirectional(
            left_rank=left_rank,
            right_rank=right_rank,
            tensor_to_left=tensor_to_left,
            tensor_to_right=tensor_to_right,
            group=group,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        return (None, None, None) + \
            NeighbourExchangeBidirectional.apply(
                ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
            )


def neighbour_exchange_bidirectional_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidirectional.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) -
    https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={
          Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas
      },
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        temperature: float = 0.05,
        logit_bias: Optional[torch.Tensor] = None,
        bidirectional: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        self._rank = rank
        self._world_size = world_size
        self._logit_scale = torch.exp(torch.log(torch.tensor([1 / temperature])))
        self._logit_bias = logit_bias
        self._bidirectional = bidirectional

    @staticmethod
    def sigmoid_loss(
        moda_features: torch.Tensor,
        modb_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        negative_only: bool = False,
    ):
        device = moda_features.device
        dtype = moda_features.dtype
        batchsize = moda_features.shape[0]

        logits = logit_scale * moda_features @ modb_features.T
        if logit_bias is not None:
            logits += logit_bias

        labels = -torch.ones((batchsize, batchsize), device=device, dtype=dtype)
        if not negative_only:
            labels += 2 * torch.eye(batchsize, device=device, dtype=dtype)

        return -f.logsigmoid(labels * logits).sum() / moda_features.shape[0]

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale
        logit_bias = logit_bias or self._logit_bias

        # we rename left and right features to modality A and modality B
        # as to not confuse with the left and right device directions
        a_features = left_features
        b_features = right_features

        loss = self.sigmoid_loss(a_features, b_features, logit_scale, logit_bias)

        if self._world_size > 1:

            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self._rank + 1) % self._world_size
            left_rank = (self._rank - 1 + self._world_size) % self._world_size

            if self._bidirectional:

                b_features_to_right = b_features_to_left = b_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    b_features_received = neighbour_exchange_bidirectional_with_grad(
                        left_rank=left_rank,
                        right_rank=right_rank,
                        tensor_to_left=b_features_to_left,
                        tensor_to_right=b_features_to_right,
                    )
                    for feats in b_features_received:
                        loss += self.sigmoid_loss(
                            a_features,
                            feats,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    b_features_to_left, b_features_to_right = b_features_received

                if remainder:
                    b_features_from_left = neighbour_exchange_with_grad(
                        from_rank=left_rank,
                        to_rank=right_rank,
                        tensor=b_features_to_right,
                    )
                    loss += self.sigmoid_loss(
                        a_features,
                        b_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                b_features_to_right = b_features
                for i in range(self.world_size - 1):
                    b_features_from_left = neighbour_exchange_with_grad(
                        from_rank=left_rank,
                        to_rank=right_rank,
                        tensor=b_features_to_right,
                    )
                    loss += self.sigmoid_loss(
                        a_features,
                        b_features_from_left,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True,
                    )
                    b_features_to_right = b_features_from_left

        return {'contrastive_loss': loss} if output_dict else loss
