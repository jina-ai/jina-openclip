from typing import Any, Dict, Optional, Set, Union

import mteb
import numpy as np
import torch
import torch.nn.functional as f
import typer
from loguru import logger
from transformers import AutoTokenizer

from open_clip import create_model_and_transforms
from open_clip.model import CLIP
from training.utils import get_autocast


class _MTEBEncoder(torch.nn.Module):
    def __init__(
        self,
        clip_model: torch.nn.Module,
        _tokenizer: Any = None,
        hf_tokenizer_name: str = '',
        batch_size: int = 4,
        max_seq_length: int = 8192,
        device: Union[str, torch.device] = 'cpu',
    ):
        super(_MTEBEncoder, self).__init__()

        self._tokenizer = None
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._device = device
        _model = clip_model
        self._model = _model

        if isinstance(_model, CLIP):
            assert _tokenizer is not None
            self._tokenizer = _tokenizer
            self._embed = self._clip_embed
        else:
            assert hf_tokenizer_name
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=hf_tokenizer_name,
                trust_remote_code=True,
                force_download=True,
            )
            self._embed = self._hf_embed

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _hf_embed(self, sentences: list[str]):
        encoded_input = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self._max_seq_length,
        ).to(self._device)

        model_output = self._model.text.transformer(**encoded_input)
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input['attention_mask']
        )
        sentence_embeddings = f.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.to(torch.float32).cpu().numpy()

    def _clip_embed(self, sentences: list[str]):
        x = self._tokenizer(sentences).to(self._device)
        sentence_embeddings = self._model.encode_text(x)
        return sentence_embeddings.to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def encode(self, sentences: list[str], batch_size: int = 1, **_):
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i: i + batch_size]
                embeddings.append(self._embed(batch))

        return np.concatenate(embeddings, axis=0)


def _filter_metrics(_metrics: Dict[str, float], select_metrics: Set[str]):
    _filtered_metrics = {}
    for key, value in _metrics.items():
        if len(select_metrics) > 0 and key not in select_metrics:
            continue
        if isinstance(value, float):
            _filtered_metrics[key] = value

    return _filtered_metrics


def main(
    model_name: str,
    pretrained: Optional[str] = None,
    hf_tokenizer_name: Optional[str] = None,
    benchmark_name: str = 'MTEB(eng)',
    task_types: str = 'Retrieval,STS',
    precision: str = 'fp16',
    batch_size: int = 4,
    max_sequence_length: int = 512,
    rank: int = 0,
    world_size: int = 1,
):
    logger.info('--------------------------------------------------------------------')
    logger.info('Loading the model ...')

    device = f'cuda:{rank}'
    model, *_ = create_model_and_transforms(
        model_name, pretrained, precision=precision, device=device,
    )
    _mteb_model = _MTEBEncoder(
        clip_model=model,
        hf_tokenizer_name=hf_tokenizer_name,
        max_seq_length=max_sequence_length,
        device=device,
    )

    logger.info('Benchmark configuration ...')
    benchmark = mteb.get_benchmark(benchmark_name)
    task_types = task_types.split(',')
    tasks = [
        task for task in benchmark.tasks if task.metadata.type in task_types
    ]
    treccovid = None
    for task in tasks:
        if task.metadata.name == 'TRECCOVID':
            treccovid = task
    if treccovid is not None:
        tasks = [treccovid] + [
            task for task in tasks if task.metadata.name != 'TRECCOVID'
        ]

    tasks_per_gpu = len(tasks) // world_size
    start = rank * tasks_per_gpu
    end = (rank + 1) * tasks_per_gpu
    selected_tasks = tasks[start:] if rank == world_size else tasks[start:end]

    autocast = get_autocast(precision)

    logger.info('Starting the MTEB benchmark ...')

    with autocast():
        evaluation = mteb.MTEB(tasks=selected_tasks)
        evaluation.run(
            model=_mteb_model,
            verbosity=2,
            encode_kwargs={'batch_size': batch_size},
            output_folder='results',
            ignore_identical_ids=False,
        )

    logger.info('Finished MTEB benchmark!')
    logger.info('--------------------------------------------------------------------')


if __name__ == '__main__':
    typer.run(main)
