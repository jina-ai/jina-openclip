import glob
import logging
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time
from contextlib import suppress
from typing import Optional

import fsspec
import numpy as np
import torch
from loguru import logger


def setup_logging(
    logfile: Optional[str] = None,
    level: int = logging.INFO,
    rank: int = 0,
    include_host: bool = False,
    include_rank: bool = False,
):
    extra = {'host': 'localhost', 'rank': 0}
    if include_host:
        import socket

        extra['host'] = socket.gethostname()

    if include_rank:
        extra['rank'] = rank

    fmt = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | '
        '<green>host={extra[host]} rank={extra[rank]}</green> | '
        '<level>{message}</level>'
    )
    logger.configure(extra=extra)
    logger.remove()
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    if logfile:
        logger.add(logfile, format=fmt, level=level)


def get_autocast(precision: str):
    if (
        precision == 'amp'
        or precision == 'float16'
        or precision == 'fp16'
        or precision == 'pure_fp16'
    ):
        return torch.cuda.amp.autocast
    elif (
        precision == 'amp_bfloat16'
        or precision == 'amp_bf16'
        or precision == 'bfloat16'
        or precision == 'bf16'
        or precision == 'pure_bf16'
    ):
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def _remote_sync_s3(localdir: str, remotedir: str) -> bool:
    result = subprocess.run(
        ['aws', 's3', 'sync', localdir, remotedir, '--exclude', '*epoch_latest.pt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logger.error(
            f'Error: Failed to sync with S3 bucket {result.stderr.decode("utf-8")}'
        )
        return False

    logger.info('Successfully synced with S3 bucket')
    return True


def _remote_sync_fsspec(localdir: str, remotedir: str) -> bool:
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(localdir)
    b = fsspec.get_mapper(remotedir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        logger.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            logger.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            logger.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            logger.info(f'Error during remote sync for {k}: {e}')
            return False

    return True


def remote_sync(localdir: str, remotedir: str, protocol: str = 's3') -> bool:
    logger.info('Starting remote sync ...')
    if protocol == 's3':
        return _remote_sync_s3(localdir, remotedir)
    elif protocol == 'fsspec':
        return _remote_sync_fsspec(localdir, remotedir)
    else:
        logger.error(f'Unknown remote protocol {protocol}')
        return False


def keep_running_remote_sync(
    sync_every: int, localdir: str, remotedir: str, protocol: str = 's3'
):
    while True:
        time.sleep(sync_every)
        remote_sync(localdir, remotedir, protocol)


def start_sync_process(
    sync_every: int,
    localdir: str,
    remotedir: str,
    protocol: str = 's3',
):
    return multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, localdir, remotedir, protocol),
    )


def pytorch_save(ptobj: object, filepath: str):
    of = fsspec.open(filepath, 'wb')
    with of as _:
        torch.save(ptobj, filepath)


def pytorch_load(filepath: str, map_location: Optional[str] = None):
    if filepath.startswith('s3'):
        logger.info('Loading remote checkpoint, which may take a bit ...')
    of = fsspec.open(filepath, 'rb')
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def check_exists(filepath: str) -> bool:
    try:
        with fsspec.open(filepath):
            pass
    except FileNotFoundError:
        return False
    return True


def random_seed(seed: int = 42, rank: int = 0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _natural_key(string_: str):
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
        checkpoints = sorted(checkpoints, key=_natural_key)
        return checkpoints[-1]
    return None


def copy_codebase(directory: str, name: str):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(directory, name, 'code')
    if os.path.exists(new_code_path):
        print(
            f'Error. Experiment already exists at {new_code_path}. '
            f'Use --name to specify a new experiment.'
        )
        return -1

    print(f'Copying codebase to {new_code_path} ...')
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb')
    )
    print('Done copying code')
    return 1
