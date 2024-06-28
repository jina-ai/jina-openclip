import logging
import math
import time
import warnings
from collections import defaultdict
from itertools import islice

import torch
from torch import nn

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip.loss import GatherFeatures

from .distributed import is_master
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        'image_features': model_out[0],
        'text_features': model_out[1],
        'logit_scale': model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, model, scaler=None, deepspeed=False):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    elif deepspeed:
        model.backward(total_loss)
    else:
        total_loss.backward()


class DummyEmbeddingsDataloader:

    def __iter__(self):
        return self

    def __next__(self):
        return None, (None, None)


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    emb_dataloader=None,
    emb_losses=None,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    if args.mtl:
        assert emb_dataloader is not None
        assert emb_losses is not None
    else:
        emb_dataloader = DummyEmbeddingsDataloader()

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    accum_images, accum_texts, accum_features = [], [], {}
    accum_emb_datasets, accum_emb_batches, accum_emb_labels, accum_embeddings = (
        [], [], [], []
    )

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    embeddings_gather = None
    if args.emb_global_batch:
        embeddings_gather = GatherFeatures(
            local_loss=False,
            gather_with_grad=args.gather_with_grad,
            rank=args.rank,
            world_size=args.world_size,
        )

    start = time.time()

    # training loop
    for i, (mm_batch, (emb_dataset, (emb_batch, emb_labels))) in enumerate(zip(
        dataloader, islice(emb_dataloader, 1, None)
    )):

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = mm_batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        if emb_batch:
            for batch in emb_batch:
                batch.to(device=device)
        if emb_labels:
            emb_labels[0] = [label.to(device=device) for label in emb_labels[0]]

        data_time_m.update(time.time() - start)

        if args.deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        if args.accum_freq == 1:

            # WITHOUT Gradient Accumulation

            with autocast():

                if args.unify_batch:
                    modelout = model(images, texts)
                    logit_scale = modelout['logit_scale']
                    embeddings = [
                        model.module.encode_text(
                            embedding['input_ids'], normalize=True
                        )
                        if isinstance(model, nn.parallel.DistributedDataParallel)
                        else model.encode_text(
                            embedding['input_ids'], normalize=True
                        )
                        for embedding in emb_batch
                    ]
                    left, right = embeddings[0], embeddings[1]
                    modelout['image_features'] = torch.cat(
                        [modelout['image_features'], left], dim=0
                    )
                    modelout['text_features'] = torch.cat(
                        [modelout['text_features'], right], dim=0,
                    )
                    losses = loss(**modelout, output_dict=True)

                else:
                    modelout = model(images, texts)
                    logit_scale = modelout['logit_scale']
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        modelout.update(
                            {f'dist_{k}': v for k, v in dist_model_out.items()}
                        )
                    losses = loss(**modelout, output_dict=True)

                    if args.mtl:
                        emb_loss_fn = (
                            emb_losses[emb_dataset]
                            if emb_dataset in emb_losses else emb_losses['*']
                        )
                        embeddings = [
                            model.module.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            if isinstance(model, nn.parallel.DistributedDataParallel)
                            else model.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            for embedding in emb_batch
                        ]
                        if args.emb_global_batch:
                            assert len(emb_labels) == 0, (
                                'Global batch cannot be used in conjunction with '
                                'labeled data'
                            )
                            all_embeddings = [
                                embeddings_gather(emb) for emb in embeddings
                            ]
                            embedding_loss = emb_loss_fn(*all_embeddings)
                        else:
                            embedding_loss = emb_loss_fn(*embeddings, *emb_labels)

                        losses['embedding_loss'] = args.emb_loss_weight * embedding_loss

            total_loss = sum(losses.values())
            losses['loss'] = total_loss
            backward(total_loss, model, scaler=scaler, deepspeed=args.deepspeed)

        else:

            # WITH Gradient Accumulation

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    if args.unify_batch:
                        modelout = model(images, texts)
                        embeddings = [
                            model.module.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            if isinstance(model, nn.parallel.DistributedDataParallel)
                            else model.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            for embedding in emb_batch
                        ]
                        left, right = embeddings[0], embeddings[1]
                        for f in ('logit_scale', 'logit_bias'):
                            modelout.pop(f, None)

                        for key, val in list(modelout.items()) + [
                            ('left_features', left), ('right_features', right)
                        ]:
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]
                    else:
                        modelout = model(images, texts)

                        for f in ('logit_scale', 'logit_bias'):
                            modelout.pop(f, None)

                        for key, val in modelout.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                        if args.mtl:
                            # if we have no labels == pair training
                            if len(emb_labels) == 0:
                                embeddings = [
                                    model.module.encode_text(
                                        embedding['input_ids'], normalize=True
                                    )
                                    if isinstance(
                                        model, nn.parallel.DistributedDataParallel
                                    )
                                    else model.encode_text(
                                        embedding['input_ids'], normalize=True
                                    )
                                    for embedding in emb_batch
                                ]
                                accum_embeddings.append(embeddings)
                            # else == triplet training
                            else:
                                accum_emb_labels.append(emb_labels)

                accum_images.append(images)
                accum_texts.append(texts)
                if args.mtl:
                    accum_emb_datasets.append(emb_dataset)
                    accum_emb_batches.append(emb_batch)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            if len(accum_emb_labels) not in {0, args.accum_freq}:
                warnings.warn(
                    f'Out of {args.accum_freq} gradient accumulation steps, '
                    f'{len(accum_emb_labels)} contain labels and '
                    f'{len(accum_embeddings)} are aligned pairs. '
                    f'Gradient accumulation cannot work with inconsistent data'
                )
                accum_images, accum_texts, accum_features = [], [], {}
                (
                    accum_emb_datasets,
                    accum_emb_batches,
                    accum_emb_labels,
                    accum_embeddings
                ) = [], [], [], []
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features
            # from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            if args.deepspeed:
                model.zero_grad()
                model.micro_steps = 0
            else:
                optimizer.zero_grad()

            losses = defaultdict(lambda: torch.zeros(1, device=device))

            for k in range(args.accum_freq):
                with autocast():
                    if args.unify_batch:
                        images = accum_images[k]
                        texts = accum_texts[k]
                        modelout = model(images, texts)
                        embbatch = accum_emb_batches[k]
                        embeddings = [
                            model.module.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            if isinstance(model, nn.parallel.DistributedDataParallel)
                            else model.encode_text(
                                embedding['input_ids'], normalize=True
                            )
                            for embedding in embbatch
                        ]
                        left, right = embeddings[0], embeddings[1]
                        modelout['left_features'] = left
                        modelout['right_features'] = right

                        inputs_no_accum = {}
                        inputs_no_accum['logit_scale'] = logit_scale = modelout.pop(
                            'logit_scale'
                        )
                        if 'logit_bias' in modelout:
                            inputs_no_accum['logit_bias'] = modelout.pop('logit_bias')

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(
                                accumulated[:k] + [modelout[key]] + accumulated[k+1:]
                            )

                        inputs['image_features'] = torch.cat(
                            [inputs['image_features'], inputs['left_features']], dim=0
                        )
                        inputs['text_features'] = torch.cat(
                            [inputs['text_features'], inputs['right_features']], dim=0,
                        )
                        _ = inputs.pop('left_features')
                        _ = inputs.pop('right_features')

                        _losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        contrastive_loss = sum(_losses.values())
                        losses['contrastive_loss'] += contrastive_loss
                        total_loss = contrastive_loss

                    else:

                        images = accum_images[k]
                        texts = accum_texts[k]

                        modelout = model(images, texts)

                        inputs_no_accum = {}
                        inputs_no_accum['logit_scale'] = logit_scale = modelout.pop(
                            'logit_scale'
                        )
                        if 'logit_bias' in modelout:
                            inputs_no_accum['logit_bias'] = modelout.pop('logit_bias')

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(
                                accumulated[:k] + [modelout[key]] + accumulated[k+1:]
                            )

                        _losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        contrastive_loss = sum(_losses.values())
                        losses['contrastive_loss'] += contrastive_loss
                        total_loss = contrastive_loss

                        if args.mtl:
                            emb_dataset = accum_emb_datasets[k]
                            emb_batch = accum_emb_batches[k]
                            emb_loss_fn = (
                                emb_losses[emb_dataset]
                                if emb_dataset in emb_losses else emb_losses['*']
                            )
                            embeddings = [
                                model.module.encode_text(
                                    embedding['input_ids'], normalize=True
                                )
                                if isinstance(
                                    model, nn.parallel.DistributedDataParallel
                                )
                                else model.encode_text(
                                    embedding['input_ids'], normalize=True
                                )
                                for embedding in emb_batch
                            ]

                            if len(accum_emb_labels) == 0:
                                inputs = []
                                _cached_embeddings = list(zip(*accum_embeddings))
                                for idx, _cached_embedding in enumerate(
                                    _cached_embeddings
                                ):
                                    inputs.append(
                                        torch.cat(
                                            _cached_embedding[:k] +
                                            (embeddings[idx],) +
                                            _cached_embedding[k+1:]
                                        )
                                    )
                                if args.emb_global_batch:
                                    assert len(emb_labels) == 0, (
                                        'Global batch cannot be used in conjunction '
                                        'with labeled data'
                                    )
                                    all_inputs = [
                                        embeddings_gather(emb) for emb in inputs
                                    ]
                                    embedding_loss = emb_loss_fn(*all_inputs)
                                else:
                                    embedding_loss = emb_loss_fn(*inputs)
                                del inputs
                            else:
                                if args.emb_global_batch:
                                    raise ValueError(
                                        'Global batch cannot be used in conjunction '
                                        'with labeled data'
                                    )
                                emb_labels = accum_emb_labels[k]
                                embedding_loss = emb_loss_fn(*embeddings, *emb_labels)

                            embedding_loss = args.emb_loss_weight * embedding_loss
                            losses['embedding_loss'] += embedding_loss
                            total_loss += embedding_loss

                losses['loss'] += total_loss
                backward(total_loss, model, scaler=scaler, deepspeed=args.deepspeed)

            losses['contrastive_loss'] = losses['contrastive_loss'] / args.accum_freq
            losses['embedding_loss'] = losses['embedding_loss'] / args.accum_freq
            losses['loss'] = losses['loss'] / args.accum_freq

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        elif args.deepspeed:
            model.step()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}
            (
                accum_emb_datasets,
                accum_emb_batches,
                accum_emb_labels,
                accum_embeddings
            ) = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - start)
        start = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = ' - '.join(
                [
                    f'{loss_name.replace("_", " ")}: '
                    f'{loss_m.val:#.5g} (avg {loss_m.avg:#.5g})'
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f'Epoch: {epoch} [{num_samples:>{sample_digits}}/'
                f'{samples_per_epoch} ({percent_complete:.0f}%)] - '
                f'Data Time: {data_time_m.avg:.3f}s - '
                f'Batch Time: {batch_time_m.avg:.3f}s - '
                f'Samples per Second: {samples_per_second:#g}/s, '
                f'{samples_per_second_per_gpu:#g}/s/gpu - '
                f'Last Layer LR: {optimizer.param_groups[-1]["lr"]:5f} - '
                f'Logit Scale: {logit_scale_scalar:.3f} - '
                f'LOSS | {loss_log}'
            )

            # Save train loss / etc. Using non avg meter values as loggers have
            # their own smoothing
            logdata = {
                'data_time': data_time_m.val,
                'batch_time': batch_time_m.val,
                'samples_per_second': samples_per_second,
                'samples_per_second_per_gpu': samples_per_second_per_gpu,
                'scale': logit_scale_scalar,
            }
            logdata.update(
                {
                    f'lr/{pgroup["###logging_descriptor"]}': pgroup['lr']
                    for pgroup in optimizer.param_groups
                }
            )
            logdata.update({name: val.val for name, val in losses_m.items()})

            logdata = {'train/' + name: val for name, val in logdata.items()}

            if tb_writer is not None:
                for name, val in logdata.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                logdata['step'] = step  # for backwards compatibility
                wandb.log(logdata, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
