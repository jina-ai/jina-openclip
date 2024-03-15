import logging
import math
import time
from itertools import islice

import torch
from torch import nn

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype

from .distributed import all_gather_object, is_master
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


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def get_global_batch_grads(
    embedding_batches: list[torch.Tensor],
    loss_fn: nn.Module,
    args,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Gather embeddings from all devices and compute gradients w.r.t. global batch.

    This function will gather the embeddings from all devices and evaluate `loss_fn` on
    *all* embeddings jointly. Then, we backpropagate the loss onto the gathered
    embeddings and return the gradients belonging to the embeddings of the local
    device. This only works for loss functions not requiring "labels". I.e., this won't
    work for distillation losses, hard negative losses with ragged input, etc.

    :param embedding_batches: List of embedding tensors,
        e.g. [left_embeddings, right_embeddings]
    :param loss_fn: Module that would accept `*embedding_batches`
    :param args: Arguments
    :return: Tuple of loss and its gradient
        The `loss_fn` is differentiated w.r.t. the tensors in `embedding_batches`
        after computing the loss over embeddings gathered from all devices.
    """
    global_embedding_batches = all_gather_object(args, embedding_batches)
    # flatten the device dimension
    global_embedding_batches = [
        b.reshape((-1, *b.shape[2:])).requires_grad_() for b in global_embedding_batches
    ]
    loss = loss_fn(*global_embedding_batches)
    loss.backward()
    # only take the gradients that belong to embeddings from this device
    grads = [
        b.grad.reshape((args.world_size, -1, *b.shape[1:]))[args.rank]
        for b in global_embedding_batches
    ]
    return grads, loss


def get_surrogate_loss(
    embeddings: list[torch.Tensor], gradients: list[torch.Tensor],
):
    """
    Perform backward pass given embeddings and corresponding gradients.

    :param embeddings: A list of embeddings, e.g. `[left, right]`
    :param gradients: Gradients of a loss functions w.r.t. `embeddings`
    :return:
    """
    flat_embeddings = torch.cat([o.flatten() for o in embeddings])
    flat_grads = torch.cat([g.flatten() for g in gradients])
    return torch.dot(flat_embeddings, flat_grads)


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
    end = time.time()

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
            emb_labels = emb_labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()

        if args.accum_freq == 1:

            # WITHOUT Gradient Accumulation

            with autocast():

                model_out = model(images, texts)
                logit_scale = model_out['logit_scale']
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update(
                        {f'dist_{k}': v for k, v in dist_model_out.items()}
                    )
                losses = loss(**model_out, output_dict=True)
                #contrastive_loss = sum(losses.values())

                #losses['contrastive_loss'] = contrastive_loss
                #total_loss = contrastive_loss

                if args.mtl:
                    emb_loss_fn = (
                        emb_losses[emb_dataset]
                        if emb_dataset in emb_losses else emb_losses['*']
                    )

                    embeddings = (
                        [
                            model.module.encode_text(embedding['input_ids'])
                            if isinstance(model, nn.parallel.DistributedDataParallel)
                            else model.encode_text(embedding['input_ids'])
                            for embedding in emb_batch
                        ]
                    )
                    if args.emb_global_batch:
                        assert len(emb_labels) == 0
                        grads, _ = get_global_batch_grads(
                            embeddings, emb_loss_fn, args
                        )
                        embedding_loss = get_surrogate_loss(embeddings, grads)
                    else:
                        embedding_loss = emb_loss_fn(*embeddings, *emb_labels)

                    losses['embedding_loss'] = args.emb_loss_weight * embedding_loss
                    # total_loss += args.emb_loss_weight * embedding_loss

            total_loss = sum(losses.values())
            losses['loss'] = total_loss
            backward(total_loss, scaler)

        else:

            # WITH Gradient Accumulation

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ('logit_scale', 'logit_bias'):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                    if args.mtl:
                        embeddings = model(emb_batch)
                        for j, embedding in enumerate(embeddings):
                            if len(accum_embeddings) == 0:
                                accum_embeddings.append([embedding])
                            else:
                                accum_embeddings[j].append(embedding)

                accum_images.append(images)
                accum_texts.append(texts)
                if args.mtl:
                    accum_emb_datasets.append(emb_dataset)
                    accum_emb_labels.append(emb_labels)
                    accum_emb_batches.append(emb_batch)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features
            # from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()

            for k in range(args.accum_freq):
                images = accum_images[k]
                texts = accum_texts[k]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum['logit_scale'] = logit_scale = model_out.pop(
                        'logit_scale'
                    )
                    if 'logit_bias' in model_out:
                        inputs_no_accum['logit_bias'] = model_out.pop('logit_bias')

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:k] + [model_out[key]] + accumulated[k+1:]
                        )

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    contrastive_loss = sum(losses.values())
                    losses['contrastive_loss'] = contrastive_loss
                    total_loss = contrastive_loss

                    if args.mtl:
                        _emb_dataset = accum_emb_datasets[k]
                        _emb_batch = accum_emb_batches[k]
                        _emb_loss_fn = (
                            emb_losses[_emb_dataset]
                            if emb_dataset in emb_losses else emb_losses['*']
                        )
                        _embeddings = model(_emb_batch)

                        inputs = []
                        for val in accum_embeddings:
                            inputs.append(
                                torch.cat(val[:k] + [_embeddings] + val[k+1:])
                            )
                        labels = torch.cat(accum_emb_labels)

                        if args.emb_global_batch:
                            assert len(labels) == 0
                            grads, _ = get_global_batch_grads(
                                inputs, emb_loss_fn, args
                            )
                            embedding_loss = get_surrogate_loss(inputs, grads)
                        else:
                            embedding_loss = emb_loss_fn(*inputs, *labels)

                        losses['embedding_loss'] = embedding_loss
                        total_loss += args.emb_loss_weight * embedding_loss

                backward(total_loss, scaler)

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

        batch_time_m.update(time.time() - end)
        end = time.time()
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
            loss_log = ' '.join(
                [
                    f'{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})'
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
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/"
                f"{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                f"{samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[-1]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
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
