from typing import Union

import numpy as np
import torch
from training.data.embeddings.utils import lookahead
from training.embloss import InputType
from transformers.tokenization_utils import PreTrainedTokenizer


@lookahead
def dynamic_collate(
    batch: tuple[
        list[str],
        list[tuple[tuple[str], Union[tuple[float], None]]],
    ],
    tokenizer: PreTrainedTokenizer,
    tokenizer_options: dict,
    input_type_dict: dict[str, str],
):
    dataset, batch = list(zip(*batch))
    dataset = dataset[0]
    input_type = (
        input_type_dict[dataset] if dataset in input_type_dict else input_type_dict['*']
    )

    if input_type in (
        InputType.MULTIPLE_NEGATIVES,
        InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
    ):
        return dataset, flexible_collate_fn(batch, tokenizer, tokenizer_options)
    else:
        return dataset, collate_fn(batch, tokenizer, tokenizer_options)


def collate_fn(
    batch: list[tuple[tuple[str], Union[tuple[float], None]]],
    tokenizer: PreTrainedTokenizer,
    tokenizer_options: dict,
):
    features = [
        tokenizer.batch_encode_plus(
            list(single_str_batch),
            **tokenizer_options,
        )
        for single_str_batch in zip(*[texts for texts, scores in batch])
    ]
    scores = [
        torch.tensor(score_batch)
        for score_batch in zip(
            *[scores for texts, scores in batch if scores is not None]
        )
    ]
    return features, scores


def flexible_collate_fn(
    batch: list[tuple[tuple[str], Union[tuple[float], None]]],
    tokenizer: PreTrainedTokenizer,
    tokenizer_options: dict,
):
    # extract all non-emtpy text values into one single list
    texts = [text for row, scores in batch for text in row if len(text) > 0]
    # determine the number of text values per row
    row_sizes = np.array(
        [len([text for text in row if len(text) > 0]) for row, scores in batch]
    )
    # tokenize the text values to construct input features for the transformer model
    features = [
        tokenizer.batch_encode_plus(
            texts,
            **tokenizer_options,
        )
    ]
    # extract the cross encoder scores for each text pair
    scores = [
        torch.tensor(s)
        for (texts, scores), r in zip(batch, row_sizes)
        for s in [elem for elem in scores][: (r - 1)]
    ]
    return features, (scores, row_sizes)
