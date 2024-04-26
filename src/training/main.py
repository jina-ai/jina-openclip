import glob
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_clip import (
    create_loss,
    create_model_and_transforms,
    get_tokenizer,
    trace_model,
)
from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH

from training.data import MultiS3EmbeddingDataset, dynamic_collate, get_multimodal_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.eval import evaluate
from training.fileutils import pt_load, remote_sync, start_sync_process
from training.logger import setup_logging
from training.optimizer import create_optimizer
from training.params import parse_args
from training.scheduler import create_scheduler
from training.train import train_one_epoch

LATEST_CHECKPOINT_NAME = 'epoch_latest.pt'


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See https://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across
    # multiple sub-folders
    if remote:
        result = subprocess.run(
            ['aws', 's3', 'ls', path + '/'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(' ')[-1])
            for x in result.stdout.decode().split('\n')[:-1]
        ]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def create_embeddings_dataloader(args):

    import training.embloss as embeddings_loss_module

    embeddings_dataset = None

    try:
        loss_init = json.loads(args.emb_losses)
    except json.JSONDecodeError:
        loss_init = [{'name': loss} for loss in args.emb_losses.split(',')]

    loss_init = loss_init or [{'name': 'InfoNCELoss'}]

    for d in loss_init:
        d['name'] = getattr(embeddings_loss_module, d['name'])
        if 'tasks' not in d:
            d['tasks'] = '*'
        if 'options' not in d:
            d['options'] = {}

    task_dict = {
        task: d['name'](**d['options']) for d in loss_init for task in d['tasks']
    }
    input_type_dict = {k: v.input_type for (k, v) in task_dict.items()}

    for loss_fn in task_dict.values():
        logging.info(f'Setting up loss: {loss_fn.__class__.__name__}')

    logging.info('Building embedding dataset ...')
    if args.resume:
        embeddings_dataset_checkpoint = os.path.join(
            args.resume, f'worker{args.rank}-dataset.json'
        )
        if os.path.isfile(embeddings_dataset_checkpoint):
            logging.info(
                f'Loading from checkpoint {embeddings_dataset_checkpoint} ...'
            )
            embeddings_dataset = MultiS3EmbeddingDataset.load_from_json(
                embeddings_dataset_checkpoint,
                world_size=args.world_size,
                global_rank=args.rank,
            )
    if embeddings_dataset is None:
        logging.info(
            f'Bucket: {args.emb_datasets_s3_bucket}, Datasets: {args.emb_datasets}'
        )
        datasets = args.emb_datasets.split(',')
        sampling_rates = [float(v) for v in args.emb_sampling_rates.split(',')]
        sampling_rates = {
            dataset: sr for dataset, sr in zip(datasets, sampling_rates)
        }
        embeddings_dataset = MultiS3EmbeddingDataset(
            bucket=args.emb_datasets_s3_bucket,
            batch_size=args.emb_batch_size,
            input_type_dict=input_type_dict,
            datasets=datasets,
            max_shards=args.emb_max_shards,
            world_size=args.world_size,
            global_rank=args.rank,
            sampling_rates=sampling_rates,
            num_batches=args.emb_num_batches,
            max_batches=args.emb_max_batches,
            seed=args.seed,
            synchronous=args.emb_global_batch,
        )

    logging.info('Setting up the embedding dataloader')

    embeddings_tokenizer = AutoTokenizer.from_pretrained(
        args.emb_tokenizer_name, force_download=True
    )
    embeddings_dataloader = DataLoader(
        dataset=embeddings_dataset,
        collate_fn=partial(
            dynamic_collate,
            tokenizer=embeddings_tokenizer,
            tokenizer_options={
                'padding': 'max_length',
                'truncation': True,
                'max_length': (
                    args.emb_max_sequence_length
                    or args.max_sequence_length
                    or DEFAULT_CONTEXT_LENGTH
                ),
                'return_tensors': 'pt',
            },
            input_type_dict=input_type_dict,
        ),
        batch_size=args.emb_batch_size,
        pin_memory=True,
    )

    return embeddings_dataset, embeddings_dataloader, task_dict


def main(args):
    args, dsinit = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use,
        # easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        datestr = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        if args.distributed:
            # sync date_str from master to all ranks
            datestr = broadcast_object(args, datestr)
        args.name = '-'.join(
            [
                datestr,
                f'model_{model_name_safe}',
                f'lr_{args.lr}',
                f'b_{args.batch_size}',
                f'j_{args.workers}',
                f'p_{args.precision}',
            ]
        )

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                'Error. Experiment already exists. Use --name {} to '
                'specify a new experiment.'
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, 'checkpoints')
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(log_base_path, 'tensorboard') if args.tensorboard else ''
        )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of
        # the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, 'checkpoints')
            if args.save_most_recent:
                print(
                    'Error. Cannot use save-most-recent with remote_sync and '
                    'resume latest.'
                )
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1

        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system
            # is under stress, however it's very difficult to fully work around such
            # situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(
                    checkpoint_path, remote=args.remote_sync is not None
                )
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every
        # args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.'
        )

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. '
            f'Device: {args.device}. Process (global: {args.rank}, local '
            f'{args.local_rank}), total {args.world_size}.'
        )
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. '
            f'Device: {args.device}. Process (global: {args.rank}, local '
            f'{args.local_rank}), total {args.world_size}.'
        )
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = (
        args.distill_model is not None and args.distill_pretrained is not None
    )
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        pretrained_hf=args.hf_load_pretrained,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from
        #  has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print(
            '=> using a layer from bitsandbytes.\n'
            '   this is an experimental feature which requires two extra pip installs\n'
            '   pip install bitsandbytes triton'
            '   please make sure to use triton 2.0.0'
        )
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear

        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(
            bnb.nn.triton_based_modules, args.use_bnb_linear
        )
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    params_file = ''
    if is_master(args):
        logging.info('Model:')
        logging.info(f'{str(model)}')
        logging.info('Params:')
        params_file = os.path.join(args.logs, args.name, 'params.txt')
        with open(params_file, 'w') as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f'  {name}: {val}')
                f.write(f'{name}: {val}\n')

    if args.distributed and not args.horovod and not args.deepspeed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args
        )

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(
                dist_model, device_ids=[device], **ddp_args
            )

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == 'synthetic':
        assert not args.trace, 'Cannot train with traced model'
        model, optimizer, scaler = create_optimizer(
            args=args, model=model, dsinit=dsinit
        )

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(
                        args.resume, tag=f'epoch_{latest_ckpt}'
                    )
                    logging.info(
                        f'=> resuming checkpoint \'{args.resume}\' '
                        f'(epoch {latest_ckpt})'
                    )
                else:
                    logging.info(f'=> no checkpoint found at \'{args.resume}\'')
            else:
                logging.info(f'=> \'{args.resume}\' does not exist!')
        else:
            checkpoint = pt_load(
                os.path.join(args.resume, 'state.pt'), map_location='cpu'
            )
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint['epoch']
                sd = checkpoint['state_dict']
                if (
                    not args.distributed
                    and next(iter(sd.items()))[0].startswith('module')
                ):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    checkpoint['optimizer']['param_groups'] = optimizer.state_dict()[
                        'param_groups'
                        ]
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(
                    f'=> resuming checkpoint \'{args.resume}\' (epoch {start_epoch})'
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f'=> loaded checkpoint \'{args.resume}\' (epoch {start_epoch})'
                )

    # initialize datasets
    # multimodal
    tokenizer = get_tokenizer(
        args.model,
        context_length=args.max_sequence_length,
        permute_start_positions=True,
    )
    data = get_multimodal_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), (
        'At least one train or eval dataset must be specified.'
    )

    emb_dataset, emb_dataloader, emb_losses = None, None, None
    if args.mtl:
        emb_dataset, emb_dataloader, emb_losses = create_embeddings_dataloader(args)

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (
            data['train'].dataloader.num_batches // args.accum_freq
        ) * args.epochs
        cooldown_steps = None
        if args.epochs_cooldown is not None:
            cooldown_steps = (
                data['train'].dataloader.num_batches // args.accum_freq
            ) * args.epochs_cooldown

        scheduler = create_scheduler(
            optimizer=optimizer,
            baselr=args.lr,
            warmup_steps=args.warmup,
            total_steps=total_steps,
            cooldown_steps=cooldown_steps,
            cooldown_power=args.lr_cooldown_power,
            cooldown_end_lr=args.lr_cooldown_end,
        )

    # determine if this worker should save logs and checkpoints. only do so if it
    # is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, 'Please install tensorboard.'
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting WandB ...')
        args.train_sz = data['train'].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data['val'].dataloader.num_samples

        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == 'latest' else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading WandB')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode

            convert_int8_model_to_inference_mode(model)

        # Evaluate.
        evaluate(
            model,
            preprocess_val,
            tokenizer,
            data,
            start_epoch,
            args,
            tb_writer=writer,
        )
        return

    if args.evaluate_on_start:
        evaluate(
            model,
            preprocess_val,
            tokenizer,
            data,
            start_epoch,
            args,
            tb_writer=writer,
        )

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            dist_model,
            args,
            emb_dataloader=emb_dataloader,
            emb_losses=emb_losses,
            tb_writer=writer,
        )
        completed_epoch = epoch + 1

        # Saving checkpoints.
        # is_master(args) can not be here while using deepspped, otherwise ckpts
        # can not be saved
        if args.logs and args.logs.lower() != 'none' and args.deepspeed:
            ds_checkpoint_path = os.path.join(args.logs, args.name, 'checkpoints')
            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                client_state = {'epoch': completed_epoch}
                model.save_checkpoint(
                    save_dir=ds_checkpoint_path,
                    tag=f'epoch_{str(completed_epoch)}',
                    client_state=client_state
                )
        elif args.save_logs:
            checkpoint_dict = {
                'epoch': completed_epoch,
                'name': args.name,
                'state_dict': original_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict['scaler'] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                ckpt_dir = f'epoch-{completed_epoch}'
                os.makedirs(
                    os.path.join(args.checkpoint_path, ckpt_dir), exist_ok=False
                )
                model_ckpt_path = os.path.join(
                    args.checkpoint_path, ckpt_dir, 'state.pt'
                )
                torch.save(checkpoint_dict, model_ckpt_path)
                if emb_dataset is not None:
                    dataset_ckpt_path = os.path.join(
                        args.checkpoint_path,
                        ckpt_dir,
                        f'worker{args.rank}-dataset.json'
                    )
                    emb_dataset.write_to_json(dataset_ckpt_path)

            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(
                    args.checkpoint_path, f'epoch-{completed_epoch - 1}'
                )
                if os.path.exists(previous_checkpoint):
                    shutil.rmtree(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, 'tmp.pt')
                latest_save_path = os.path.join(
                    args.checkpoint_path, LATEST_CHECKPOINT_NAME
                )
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        if (
            any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2'))
            or args.clip_benchmark_frequency != 0
            or args.mteb_frequency != 0
        ):
            evaluate(
                model,
                preprocess_val,
                tokenizer,
                data,
                completed_epoch,
                args,
                tb_writer=writer,
            )

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, 'code')
    if os.path.exists(new_code_path):
        print(
            f'Error. Experiment already exists at {new_code_path}. '
            f'Use --name to specify a new experiment.'
        )
        return -1
    print(f'Copying codebase to {new_code_path}')
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb')
    )
    print('Done copying code.')
    return 1


if __name__ == '__main__':
    main(sys.argv[1:])
