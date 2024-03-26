import itertools
import math
from copy import deepcopy
from enum import IntEnum

import pytorch_metric_learning.distances as pml_distances
import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn


class InputType(IntEnum):
    PAIR = 2
    TRIPLET = 3
    SCORED_TRIPLET = 4
    MULTIPLE_NEGATIVES = 5
    MULTIPLE_NEGATIVES_WITHOUT_SCORES = 6
    PAIR_WITH_SCORES = 7
    TEXT_WITH_LABEL = 8


def get_tuple_length(input_type: InputType):
    if input_type in (InputType.PAIR, InputType.PAIR_WITH_SCORES):
        return 2
    elif input_type in (InputType.TRIPLET, InputType.SCORED_TRIPLET):
        return 3
    elif input_type in (InputType.TEXT_WITH_LABEL,):
        return 1
    elif input_type in (
        InputType.MULTIPLE_NEGATIVES,
        InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
    ):
        return 9


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def mean_cosine_similarity(*embedding_batches):
    """Compute the mean cosine similarity across all embeddings.

    This method accepts any number of PyTorch tensors of shape $(B, D)$
    representing batches of (unnormalized) embeddings. We compute the
    mean cosine similarity obtained by considering all possible (ordered)
    pairs of embeddings. This can be computed in linear time by using
    the bilinearity of the scalar product.
    """
    all_embeddings = torch.cat(embedding_batches, dim=0)
    all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=1)
    mean_embedding = torch.mean(all_embeddings, dim=0)
    mean_similarity = torch.linalg.norm(mean_embedding) ** 2
    return mean_similarity


def info_nce(left, right, temperature):
    logits = nn.functional.log_softmax(cos_sim(left, right) / temperature, dim=1)
    return -torch.mean(torch.diag(logits))


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
        bidirectional: bool = True,
    ):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(self, embeddings_left, embeddings_right):
        loss = info_nce(embeddings_left, embeddings_right, self.temperature)
        if self.bidirectional:
            loss += info_nce(embeddings_right, embeddings_left, self.temperature)
            return loss / 2
        return loss

    @property
    def input_type(self):
        return InputType.PAIR


class CosineMSELoss(nn.Module):
    def __init__(self):
        super(CosineMSELoss, self).__init__()
        self._mse_loss = nn.MSELoss()

    def forward(self, embedding_u, embedding_v, labels):
        output = torch.functional.F.cosine_similarity(embedding_u, embedding_v)
        return self._mse_loss(output, labels)

    @property
    def input_type(self):
        return InputType.PAIR_WITH_SCORES


class CosinePearsonLoss(nn.Module):
    def __init__(self):
        super(CosinePearsonLoss, self).__init__()

    def forward(self, embedding_u, embedding_v, labels):
        output = torch.functional.F.cosine_similarity(embedding_u, embedding_v)
        corrcoef = torch.corrcoef(torch.cat([output[None, :], labels[None, :]]))
        return -corrcoef[0, 1]

    @property
    def input_type(self):
        return InputType.PAIR_WITH_SCORES


class ExtendedTripletLoss(nn.Module):
    def __init__(self, triplet_margin: float = 0.05, temperature: float = 0.05):
        super(ExtendedTripletLoss, self).__init__()
        self.triplet_margin = triplet_margin
        self.temperature = temperature
        self._nce_loss = NTXentLoss(
            distance=pml_distances.CosineSimilarity(), temperature=self.temperature
        )
        self._ce_loss = nn.CrossEntropyLoss()

    def forward(self, embedding_anchor, embedding_pos, embedding_neg):
        batch_size = embedding_anchor.shape[0]

        # triplet loss
        score_pos = 1.0 - torch.functional.F.cosine_similarity(
            embedding_anchor, embedding_pos
        )
        score_neg = 1.0 - torch.functional.F.cosine_similarity(
            embedding_anchor, embedding_neg
        )
        losses = torch.functional.F.relu(score_pos - score_neg + self.triplet_margin)
        triplet_loss = losses.mean()

        # info nce loss with hard negatives
        embedding_targets = torch.cat([embedding_pos, embedding_neg])
        labels = torch.arange(start=0, end=batch_size, device=embedding_anchor.device)
        scores = cos_sim(embedding_anchor, embedding_targets) / self.temperature
        info_nce_hn_loss = self._ce_loss(scores, labels)

        # info nce loss reverse
        ref_labels = deepcopy(labels)
        inf_nce_reverse_loss = self._nce_loss(
            labels=labels,
            ref_labels=ref_labels,
            embeddings=embedding_pos,
            ref_emb=embedding_anchor,
        )

        loss = (info_nce_hn_loss + inf_nce_reverse_loss + triplet_loss) / 3
        return loss

    @property
    def input_type(self):
        return InputType.TRIPLET


class CELoss(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super(CELoss, self).__init__()
        self._kl_loss = nn.KLDivLoss(reduction="batchmean")
        self._info_nce_loss = InfoNCEHardNegativeLoss()
        self._alpha = alpha

    def forward(
        self, embedding_anchor, embedding_pos, embedding_neg, scores_pos, scores_neg
    ):
        emb_scores_pos = torch.functional.F.cosine_similarity(
            embedding_anchor, embedding_pos
        )
        emb_scores_neg = torch.functional.F.cosine_similarity(
            embedding_anchor, embedding_neg
        )
        emb_scores = torch.stack([emb_scores_pos, emb_scores_neg], dim=1)
        emb_scores = torch.nn.functional.log_softmax(emb_scores, dim=1)
        ce_scores = torch.stack([scores_pos, scores_neg], dim=1)
        ce_scores = torch.nn.functional.softmax(ce_scores, dim=1)
        kl_loss = self._kl_loss(emb_scores, ce_scores)
        info_nce_loss = self._info_nce_loss(
            embedding_anchor, embedding_pos, embedding_neg
        )
        loss = kl_loss + self._alpha * info_nce_loss
        return loss

    @property
    def input_type(self):
        return InputType.SCORED_TRIPLET


class CoSentLoss(nn.Module):
    def __init__(self, tau: float = 0.05):
        """
        Computes a loss that tries to maximize the similarity of text values with the
        same label and to minimize the similarity values with different labels.

        :param tau: inverse factor to scale the similarity values
        """
        super(CoSentLoss, self).__init__()
        self._tau = tau

    def forward(self, embeddings, labels):
        # group embeddings by labels
        distinct_labels = {}
        idx = 0
        for label in labels:
            if label.item() not in distinct_labels:
                distinct_labels[label.item()] = idx
                idx += 1
        all_vectors = [[] for _ in distinct_labels]
        for label, embedding in zip(labels, embeddings):
            all_vectors[distinct_labels[label.item()]].append(embedding)
        all_vectors = [torch.stack(group) for group in all_vectors]
        sum_value = torch.tensor(1.0, requires_grad=True)
        if (
            len(all_vectors) > 1
        ):  # without negatives you can not calculate the cosent loss
            for i in range(len(all_vectors)):
                # select vectors of one row
                group_vecs = all_vectors[i]
                if (
                    len(group_vecs) < 2
                ):  # if there is only one element there are not positive pairs
                    continue
                group_vecs = torch.nn.functional.normalize(group_vecs, p=2, dim=1)
                # select remaining vectors
                other_vecs = torch.cat(all_vectors[:i] + all_vectors[i + 1:])
                other_vecs = torch.nn.functional.normalize(other_vecs, p=2, dim=1)
                pos_sim_values = (
                    group_vecs @ group_vecs.T
                )  # contains unwanted 1s in diagonal
                neg_sim_values = group_vecs @ other_vecs.T
                sum_exp_neg_sim_values = torch.sum(
                    torch.exp(neg_sim_values / self._tau)
                )
                exp_pos_sim_values = torch.exp(-pos_sim_values / self._tau)
                exp_pos_sim_values = exp_pos_sim_values * (
                    1 - torch.eye(exp_pos_sim_values.shape[0]).to(embeddings.device)
                )  # remove unwanted 1s
                sum_value = sum_value + torch.sum(
                    exp_pos_sim_values * sum_exp_neg_sim_values
                )
        loss = torch.log(sum_value)
        return loss

    @property
    def input_type(self):
        return InputType.TEXT_WITH_LABEL


class MultiCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        temperature: float = 0.05,
        bidirectional: bool = False,
        single_info_nce: bool = False,
        mem_opt: bool = False,
        chunk_size: int = 8,
    ):
        super(MultiCELoss, self).__init__()

        if bidirectional and not single_info_nce:
            raise ValueError(
                "Bidirectional loss should only be used with single info nce loss."
            )

        self._kl_loss = nn.KLDivLoss(reduction="batchmean")
        if mem_opt:
            self._info_nce_loss = InfoNCEHardNegativeLossMemOpt(
                temperature=temperature,
                bidirectional=bidirectional,
                chunk_size=chunk_size,
            )
        else:
            self._info_nce_loss = InfoNCEHardNegativeLoss(
                temperature=temperature, bidirectional=bidirectional
            )
        self._alpha = alpha
        self._beta = beta
        self._single_info_nce = single_info_nce

    def forward(self, embeddings, scores, row_sizes):
        scores_masks = row_sizes - 1
        loss = None
        batch_count = 0
        anchor_embeddings = []
        pos_embeddings = []
        neg_embeddings = []

        for emb_row, score_row in zip(
            self._iterate_batch(embeddings, row_sizes),
            self._iterate_batch(scores, scores_masks),
        ):
            pos_score = torch.functional.F.cosine_similarity(
                emb_row[0].unsqueeze(0), emb_row[1].unsqueeze(0)
            )
            neg_scores = torch.functional.F.cosine_similarity(
                emb_row[0].unsqueeze(0), emb_row[2:]
            )
            emb_scores = torch.nn.functional.log_softmax(
                torch.cat([pos_score, neg_scores]), dim=0
            )
            if self._beta > 0:
                ce_scores = torch.nn.functional.softmax(torch.stack(score_row))
                kl_loss = self._kl_loss(emb_scores, ce_scores)
            else:
                kl_loss = 0.0
            if loss is None:
                loss = self._beta * kl_loss
            else:
                loss += self._beta * kl_loss
            if not self._single_info_nce:
                loss += self._alpha * self._info_nce_loss(
                    emb_row[0].unsqueeze(0), emb_row[1].unsqueeze(0), emb_row[2:]
                )
            else:
                anchor_embeddings.append(emb_row[0].unsqueeze(0))
                pos_embeddings.append(emb_row[1].unsqueeze(0))
                neg_embeddings.append(emb_row[2:])
            batch_count += 1
        loss /= batch_count
        if self._single_info_nce:
            loss += self._alpha * self._info_nce_loss(
                torch.cat(anchor_embeddings),
                torch.cat(pos_embeddings),
                torch.cat(neg_embeddings),
            )

        return loss

    @staticmethod
    def _iterate_batch(inputs, mask):
        """Iterate through a batch row by row."""
        lower_limit = 0
        for upper_limit in itertools.accumulate(mask, lambda x, y: x + y):
            yield inputs[lower_limit:upper_limit]
            lower_limit = upper_limit

    @property
    def input_type(self):
        if self._beta != 0:
            return InputType.MULTIPLE_NEGATIVES
        else:
            return InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES


class MarginMSELoss(nn.Module):
    def __init__(self):
        super(MarginMSELoss, self).__init__()
        self._mse_loss = nn.MSELoss()

    @staticmethod
    def _dot_similarity(a: Tensor, b: Tensor):
        return (a * b).sum(dim=-1)

    def forward(
        self, embedding_anchor, embedding_pos, embedding_neg, scores_pos, scores_neg
    ):
        emb_scores_pos = self._dot_similarity(embedding_anchor, embedding_pos)
        emb_scores_neg = self._dot_similarity(embedding_anchor, embedding_neg)
        margin_emb = emb_scores_pos - emb_scores_neg
        margin_ce = scores_pos - scores_neg
        return self._mse_loss(margin_emb, margin_ce)

    @property
    def input_type(self):
        return InputType.SCORED_TRIPLET


class InfoNCEPlus(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super(InfoNCEPlus, self).__init__()
        self.temperature = temperature
        self._ce_loss = nn.CrossEntropyLoss()

    def forward(self, embedding_anchor, embedding_pos, embedding_neg):
        batch_size = embedding_anchor.shape[0]

        # info nce loss with hard negatives
        embedding_targets = torch.cat([embedding_pos, embedding_neg])
        labels = torch.arange(start=0, end=batch_size, device=embedding_anchor.device)
        scores = cos_sim(embedding_anchor, embedding_targets) / self.temperature
        info_nce_hn_loss = self._ce_loss(scores, labels)
        return info_nce_hn_loss

    @property
    def tuple_length(self):
        return 3


class InfoNCEHardNegativeLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, bidirectional: bool = False):
        super(InfoNCEHardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss()
        self._bidirectional = bidirectional

    def forward(self, embedding_anchor, embedding_pos, embedding_neg):
        batch_size = embedding_anchor.shape[0]
        embedding_targets = torch.cat([embedding_pos, embedding_neg])
        labels = torch.arange(start=0, end=batch_size, device=embedding_anchor.device)
        scores = cos_sim(embedding_anchor, embedding_targets) / self.temperature
        loss = self.loss(scores, labels)
        if self._bidirectional:
            scores = cos_sim(embedding_pos, embedding_anchor) / self.temperature
            loss += self.loss(scores, labels)
        return loss

    @property
    def input_type(self):
        return InputType.TRIPLET


class InfoNCEHardNegativeLossMemOpt(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
        bidirectional: bool = False,
        chunk_size: int = 8,
    ):
        super(InfoNCEHardNegativeLossMemOpt, self).__init__()
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss()
        self._bidirectional = bidirectional
        self._chunk_size = chunk_size

    def forward(self, embedding_anchor, embedding_pos, embedding_neg):
        batch_size = embedding_anchor.shape[0]
        embedding_targets = torch.cat([embedding_pos, embedding_neg])
        loss = 0.0
        for i in range(math.ceil(len(embedding_anchor) / self._chunk_size)):
            labels = torch.arange(
                start=i * self._chunk_size,
                end=min((i + 1) * self._chunk_size, batch_size),
                device=embedding_anchor.device,
            )
            scores = (
                cos_sim(
                    embedding_anchor[i * self._chunk_size: (i + 1) * self._chunk_size],
                    embedding_targets,
                )
                / self.temperature
            )
            partial_loss = self.loss(scores, labels) * (len(labels) / batch_size)
            if self._bidirectional:
                scores = (
                    cos_sim(
                        embedding_pos[
                            i * self._chunk_size: (i + 1) * self._chunk_size
                        ],
                        embedding_anchor,
                    )
                    / self.temperature
                )
                partial_loss += self.loss(scores, labels) * (len(labels) / batch_size)
            loss += partial_loss
        return loss

    @property
    def input_type(self):
        return InputType.TRIPLET
