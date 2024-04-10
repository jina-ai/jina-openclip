import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so 
    # horovod arg still required...
    ompi_vars = ['OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE']
    pmi_vars = ['PMI_RANK', 'PMI_SIZE']
    if all([var in os.environ for var in ompi_vars]) or all(
        [var in os.environ for var in pmi_vars]
    ):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        'LOCAL_RANK',
        'MPI_LOCALRANKID',
        'SLURM_LOCALID',
        'OMPI_COMM_WORLD_LOCAL_RANK',
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, 'Horovod is not installed'
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(seconds=3000),
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                timeout=timedelta(seconds=3000),
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    if args.horovod:
        return hvd.allgather_object(obj)
    else:
        objects = [None for _ in range(args.world_size)]
        dist.all_gather_object(objects, obj)
        return objects


def create_deepspeed_config(args):
    args.deepspeed_config = os.path.join(os.getcwd(), 'deepspeed.json')
    _, _, world_size = world_info_from_env()

    with open(args.deepspeed_config, 'w') as f:
        dsconfig = {
            'train_batch_size': args.batch_size * world_size,
            'train_micro_batch_size_per_gpu': args.batch_size,
            'gradient_accumulation_steps': args.accum_freq,
            'steps_per_print': 1000,
            'optimizer': {
                'type': 'Adam',
                'adam_w_mode': True,
                'params': {
                    'bias_correction': True,
                    'betas': [
                        args.beta1,
                        args.beta2
                    ],
                    'eps': args.eps
                }
            },
            'fp16': {
                'enabled': True,
                'loss_scale': 0,
                'initial_scale_power': 16,
                'loss_scale_window': 1000,
                'hysteresis': 2,
                'min_loss_scale': 1
            },
            # 'bf16': {
            #     'enabled': True
            # },
            'amp': {
                'enabled': False,
                'opt_level': 'O2'
            },
            'flops_profiler': {
                'enabled': True,
                'profile_step': -1,
                'module_depth': -1,
                'top_modules': 1,
                'detailed': True,
            },
        }

        if args.grad_clip_norm is not None:
            dsconfig.update({'gradient_clipping': args.grad_clip_norm})

        if args.zero_stage == 1:
            dsconfig.update(
                {
                    'zero_optimization': {
                        'stage': 1, 
                        'reduce_bucket_size': 5e8,
                        # 'offload_optimizer': {
                        #     'device': 'cpu'
                        # }
                    }
                }
            )
        elif args.zero_stage > 1:
            raise NotImplementedError()

        f.write(json.dumps(dsconfig, indent=2))

        return dsconfig
