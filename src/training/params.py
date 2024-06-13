import argparse
import ast
import os


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if 'vit' in model_name:
        return {'lr': 5.0e-4, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1.0e-6}
    else:
        return {'lr': 5.0e-4, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(
                    value
                )  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-data',
        type=str,
        default=None,
        help='Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.',
    )
    parser.add_argument(
        '--train-data-upsampling-factors',
        type=str,
        default=None,
        help=(
            'When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. '
            'Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) '
            'By default, datapoints are sampled uniformly regardless of the dataset sizes.'
        ),
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default=None,
        help='Path to file(s) with validation data',
    )
    parser.add_argument(
        '--train-num-samples',
        type=int,
        default=None,
        help='Number of samples in dataset. Required for webdataset if not available in info file.',
    )
    parser.add_argument(
        '--val-num-samples',
        type=int,
        default=None,
        help='Number of samples in dataset. Useful for webdataset if not available in info file.',
    )
    parser.add_argument(
        '--dataset-type',
        choices=['webdataset', 'csv', 'synthetic', 'auto'],
        default='auto',
        help='Which type of dataset to process.',
    )
    parser.add_argument(
        '--dataset-resampled',
        default=False,
        action='store_true',
        help='Whether to use sampling with replacement for webdataset shard selection.',
    )
    parser.add_argument(
        '--csv-separator',
        type=str,
        default='\t',
        help='For csv-like datasets, which separator to use.',
    )
    parser.add_argument(
        '--csv-img-key',
        type=str,
        default='filepath',
        help='For csv-like datasets, the name of the key for the image paths.',
    )
    parser.add_argument(
        '--csv-caption-key',
        type=str,
        default='title',
        help='For csv-like datasets, the name of the key for the captions.',
    )
    parser.add_argument(
        '--imagenet-val',
        type=str,
        default=None,
        help='Path to imagenet val set for conducting zero shot evaluation.',
    )
    parser.add_argument(
        '--imagenet-v2',
        type=str,
        default=None,
        help='Path to imagenet v2 for conducting zero shot evaluation.',
    )
    parser.add_argument(
        '--logs',
        type=str,
        default='./logs/',
        help='Where to store tensorboard logs. Use None to avoid storing logs.',
    )
    parser.add_argument(
        '--log-local',
        action='store_true',
        default=False,
        help='log files on local master, otherwise global master only.',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Optional identifier for the experiment when storing logs. Otherwise use current time.',
    )
    parser.add_argument(
        '--workers', type=int, default=4, help='Number of dataloader workers per GPU.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help='Batch size per GPU.'
    )
    parser.add_argument(
        '--epochs', type=int, default=32, help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--epochs-cooldown',
        type=int,
        default=None,
        help='When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.',
    )
    parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
    parser.add_argument(
        '--text-lr', type=float, default=None, help='Learning rate for text tower'
    )
    parser.add_argument('--beta1', type=float, default=None, help='Adam beta 1.')
    parser.add_argument('--beta2', type=float, default=None, help='Adam beta 2.')
    parser.add_argument('--eps', type=float, default=None, help='Adam epsilon.')
    parser.add_argument('--wd', type=float, default=0.2, help='Weight decay.')
    parser.add_argument(
        '--warmup', type=int, default=10000, help='Number of steps to warmup for.'
    )
    parser.add_argument(
        '--text-lr-decay',
        type=float,
        default=1.0,
        help=(
            'Layerwise Learning Rate Decay (LLRD) factor for the text tower. '
            'Value of 1.0 means no LLRD'
        ),
    )
    parser.add_argument(
        '--vision-lr-decay',
        type=float,
        default=1.0,
        help=(
            'Layerwise Learning Rate Decay (LLRD) factor for the vision tower. '
            'Value of 1.0 means no LLRD'
        ),
    )
    parser.add_argument(
        '--use-bn-sync',
        default=False,
        action='store_true',
        help='Whether to use batch norm sync.',
    )
    parser.add_argument(
        '--skip-scheduler',
        action='store_true',
        default=False,
        help='Use this flag to skip the learning rate decay.',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help=(
            "Optimizer type, one of 'adamw' or 'lamb', "
            "'lamb' is only available when using DeepSpeed"
        ),
    )
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        '--lr-cooldown-end',
        type=float,
        default=0.0,
        help='End learning rate for cooldown schedule. Default: 0',
    )
    parser.add_argument(
        '--lr-cooldown-power',
        type=float,
        default=1.0,
        help='Power for polynomial cooldown schedule. Default: 1.0 (linear decay)',
    )
    parser.add_argument(
        '--save-frequency', type=int, default=1, help='How often to save checkpoints.'
    )
    parser.add_argument(
        '--evaluate-on-start',
        action='store_true',
        default=False,
        help='Run the first benchmark evaluation before training',
    )
    parser.add_argument(
        '--save-most-recent',
        action='store_true',
        default=False,
        help='Always save the most recent model trained to epoch_latest.pt.',
    )
    parser.add_argument(
        '--zeroshot-frequency', type=int, default=2, help='How often to run zero shot.'
    )
    parser.add_argument(
        '--val-frequency',
        type=int,
        default=1,
        help='How often to run evaluation with val data.',
    )
    parser.add_argument(
        '--clip-benchmark-frequency',
        type=int,
        default=5,
        help='How often to run evaluation using the CLIP benchmark.',
    )
    parser.add_argument(
        '--mteb-frequency',
        type=int,
        default=5,
        help='How often to run evaluation on MTEB.',
    )
    parser.add_argument(
        '--resume',
        default=None,
        type=str,
        help='path to latest checkpoint (default: none)',
    )
    parser.add_argument(
        '--precision',
        choices=[
            'amp',
            'amp_bf16',
            'amp_bfloat16',
            'bf16',
            'fp16',
            'pure_bf16',
            'pure_fp16',
            'fp32',
        ],
        default='amp',
        help='Floating point precision.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='RN50',
        help='Name of the vision backbone to use.',
    )
    parser.add_argument(
        '--pretrained',
        default='',
        type=str,
        help='Use a pretrained CLIP model weights with the specified tag or file path.',
    )
    parser.add_argument(
        '--pretrained-image',
        default=False,
        action='store_true',
        help='Load imagenet pretrained weights for image tower backbone if available.',
    )
    parser.add_argument(
        '--hf-load-pretrained',
        default=True,
        action='store_false',
        help='Randomly initialize the weights for a hugging face text tower backbone if needed.',
    )
    parser.add_argument(
        '--lock-image',
        default=False,
        action='store_true',
        help='Lock full image tower by disabling gradients.',
    )
    parser.add_argument(
        '--lock-image-unlocked-groups',
        type=int,
        default=0,
        help='Leave last n image tower layer groups unlocked.',
    )
    parser.add_argument(
        '--lock-image-freeze-bn-stats',
        default=False,
        action='store_true',
        help='Freeze BatchNorm running stats in image tower for any locked layers.',
    )
    parser.add_argument(
        '--image-mean',
        type=float,
        nargs='+',
        default=None,
        metavar='MEAN',
        help='Override default image mean value of dataset',
    )
    parser.add_argument(
        '--image-std',
        type=float,
        nargs='+',
        default=None,
        metavar='STD',
        help='Override default image std deviation of of dataset',
    )
    parser.add_argument(
        '--image-interpolation',
        default=None,
        type=str,
        choices=['bicubic', 'bilinear', 'random'],
        help='Override default image resize interpolation',
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None,
        type=str,
        choices=['shortest', 'longest', 'squash'],
        help='Override default image resize (& crop) mode during inference',
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        '--max-sequence-length',
        default=None,
        type=int,
        help='CLIP training max sequence length.',
    )
    parser.add_argument(
        '--grad-checkpointing',
        default=False,
        action='store_true',
        help='Enable gradient checkpointing.',
    )
    parser.add_argument(
        '--local-loss',
        default=False,
        action='store_true',
        help='calculate loss w/ local features @ global (instead of realizing full global @ global matrix)',
    )
    parser.add_argument(
        '--gather-with-grad',
        default=False,
        action='store_true',
        help='enable full distributed gradient for feature gather',
    )
    parser.add_argument(
        '--force-image-size',
        type=int,
        nargs='+',
        default=None,
        help='Override default image size',
    )
    parser.add_argument(
        '--force-quick-gelu',
        default=False,
        action='store_true',
        help='Force use of QuickGELU activation for non-OpenAI transformer models.',
    )
    parser.add_argument(
        '--force-patch-dropout',
        default=None,
        type=float,
        help='Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper',
    )
    parser.add_argument(
        '--force-custom-text',
        default=False,
        action='store_true',
        help='Force use of CustomTextCLIP model (separate text-tower).',
    )
    parser.add_argument(
        '--torchscript',
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        '--torchcompile',
        default=False,
        action='store_true',
        help='torch.compile() the model, requires pytorch 2.0 or later.',
    )
    parser.add_argument(
        '--trace',
        default=False,
        action='store_true',
        help='torch.jit.trace the model for inference / eval only',
    )
    parser.add_argument(
        '--accum-freq',
        type=int,
        default=1,
        help='Update the model every --acum-freq steps.',
    )
    # arguments for distributed training
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up distributed training',
    )
    parser.add_argument(
        '--dist-backend', default='nccl', type=str, help='distributed backend'
    )
    parser.add_argument(
        '--report-to',
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument(
        '--wandb-notes', default='', type=str, help='Notes if logging with wandb'
    )
    parser.add_argument(
        '--wandb-project-name',
        type=str,
        default='open-clip',
        help='Name of the project if logging with wandb.',
    )
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='If true, more information is logged.',
    )
    parser.add_argument(
        '--copy-codebase',
        default=False,
        action='store_true',
        help='If true, we copy the entire base on the log directory, and execute from there.',
    )
    parser.add_argument(
        '--horovod',
        default=False,
        action='store_true',
        help='Use horovod for distributed training.',
    )
    parser.add_argument(
        '--deepspeed',
        action='store_true',
        default=False,
        help='Use deepspeed for distributed training.',
    )
    parser.add_argument(
        '--zero-stage',
        type=int,
        default=1,
        help='Stage of ZeRO algorith, applicable if deepspeed is enabled.',
    )
    parser.add_argument(
        '--zero-bucket-size',
        type=int,
        default=1e6,
        help='ZeRO algorith allgather and reduce bucket size.',
    )
    parser.add_argument(
        '--ddp-static-graph',
        default=False,
        action='store_true',
        help='Enable static graph optimization for DDP in PyTorch >= 1.11.',
    )
    parser.add_argument(
        '--no-set-device-rank',
        default=False,
        action='store_true',
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument('--seed', type=int, default=0, help='Default random seed.')
    parser.add_argument(
        '--grad-clip-norm', type=float, default=None, help='Gradient clip.'
    )
    parser.add_argument(
        '--lock-text',
        default=False,
        action='store_true',
        help='Lock full text tower by disabling gradients.',
    )
    parser.add_argument(
        '--lock-text-unlocked-layers',
        type=int,
        default=0,
        help='Leave last n text tower layer groups unlocked.',
    )
    parser.add_argument(
        '--lock-text-freeze-layer-norm',
        default=False,
        action='store_true',
        help='Freeze LayerNorm running stats in text tower for any locked layers.',
    )
    parser.add_argument(
        '--log-every-n-steps',
        type=int,
        default=100,
        help='Log every n steps to tensorboard/console/wandb.',
    )
    parser.add_argument(
        '--coca-caption-loss-weight',
        type=float,
        default=2.0,
        help='Weight assigned to caption loss in CoCa.',
    )
    parser.add_argument(
        '--coca-contrastive-loss-weight',
        type=float,
        default=1.0,
        help='Weight assigned to contrastive loss when training CoCa.',
    )
    parser.add_argument(
        '--3towers-cos-embeddings-loss-weight',
        type=float,
        default=2.0,
        help='Weight assigned to caption loss in CoCa.',
    )
    parser.add_argument(
        '--3towers-contrastive-loss-weight',
        type=float,
        default=1.0,
        help='Weight assigned to contrastive loss when training CoCa.',
    )
    parser.add_argument(
        '--remote-sync',
        type=str,
        default=None,
        help='Optinoally sync with a remote path specified by this arg',
    )
    parser.add_argument(
        '--remote-sync-frequency',
        type=int,
        default=300,
        help='How frequently to sync to a remote directly if --remote-sync is not None.',
    )
    parser.add_argument(
        '--remote-sync-protocol',
        choices=['s3', 'fsspec'],
        default='s3',
        help='How to do the remote sync backup if --remote-sync is not None.',
    )
    parser.add_argument(
        '--delete-previous-checkpoint',
        default=False,
        action='store_true',
        help='If true, delete previous checkpoint after storing a new one.',
    )
    parser.add_argument(
        '--distill-model',
        default=None,
        help='Which model arch to distill from, if any.',
    )
    parser.add_argument(
        '--distill-pretrained',
        default=None,
        help='Which pre-trained weights to distill from, if any.',
    )
    parser.add_argument(
        '--use-bnb-linear',
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.',
    )
    parser.add_argument(
        '--siglip',
        default=False,
        action='store_true',
        help='Use SigLip (sigmoid) loss.',
    )
    parser.add_argument(
        '--clip-benchmark-datasets',
        type=str,
        default='wds/mscoco_captions,wds/flickr8k,wds/flickr30k,wds/imagenetv2',
        help='Specify datasets for CLIP benchmark.',
    )
    parser.add_argument(
        '--clip-benchmark-dataset-root',
        type=str,
        default=(
            'https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}'
            '/tree/main'
        ),
        help='Specify dataset root for CLIP benchmark.',
    )
    parser.add_argument(
        '--clip-benchmark-recall-ks',
        type=str,
        default='1,5',
        help='Define a comma separated list of k values.',
    )
    parser.add_argument(
        '--mteb-tasks',
        type=str,
        default='STS12,STS15,STS17',
        help='Define a comma separated list of MTEB tasks to evaluate on.',
    )
    parser.add_argument(
        '--mteb-tokenizer-name',
        type=str,
        default='jinaai/jina-embeddings-v2-base-en',
        help='The tokenizer to use when running the MTEB benchmark.',
    )
    parser.add_argument(
        '--mteb-max-sequence-length',
        type=int,
        default=8192,
        help='The max sequence length used during MTEB evaluation.',
    )
    parser.add_argument(
        '--mtl',
        default=False,
        action='store_true',
        help='Train jointly with Multi Task Learning.',
    )
    parser.add_argument(
        '--emb-datasets',
        type=str,
        default='',
        help='Comma separated list of embedding datasets.',
    )
    parser.add_argument(
        '--emb-losses',
        type=str,
        default='',
        help='Comma separated or JSON list of loss functions to use.',
    )
    parser.add_argument(
        '--emb-datasets-s3-bucket',
        type=str,
        default='embedding-datasets',
        help='The S3 bucket with the embeddings datasets.',
    )
    parser.add_argument(
        '--emb-sampling-rates',
        type=str,
        default='',
        help='Comma separated list of sampling rates.',
    )
    parser.add_argument(
        '--emb-batch-size',
        type=int,
        default=128,
        help='The batch size of the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-tokenizer-name',
        type=str,
        default='',
        help='The tokenizer to use for the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-max-sequence-length',
        type=int,
        default=None,
        help='The max sequence length of the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-max-shards',
        type=int,
        default=None,
        help='The max shards of the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-num-batches',
        type=int,
        default=0,
        help='The num batches of the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-max-batches',
        type=int,
        default=None,
        help='The max batches of the embedding dataloader.',
    )
    parser.add_argument(
        '--emb-global-batch',
        default=False,
        action='store_true',
        help='Collect embeddings from all devices when calculating the loss.',
    )
    parser.add_argument(
        '--emb-loss-weight',
        type=float,
        default=1.0,
        help='The weighing factor for the embedding loss.',
    )
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    defaultparams = get_default_params(args.model)
    for name, val in defaultparams.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.deepspeed:
        try:
            import deepspeed

            os.environ['ENV_TYPE'] = 'deepspeed'
            dsinit = deepspeed.initialize
        except ImportError or ModuleNotFoundError:
            print("DeepSpeed is not available, please run 'pip install deepspeed'")
            exit(0)
    else:
        os.environ['ENV_TYPE'] = 'pytorch'
        dsinit = None

    return args, dsinit
