import os
import time
import argparse
import math
from numpy import finfo
import train

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from glove import create_glove_dict

def validate(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name, hparams):
    if hparams.distributed_run:
        train.init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = train.load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
     
    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    glove = None
    if hparams.encoder_conditioning:
        glove = create_glove_dict("glove_sample.txt")
    valset = TextMelLoader(hparams.validation_files, hparams,glove)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    logger = train.prepare_directories_and_logger(
        output_directory, log_directory, rank)

    if warm_start:
        model = train.warm_start_model(
            checkpoint_path, model, hparams.ignore_layers)
    else:
        model, optimizer, _learning_rate, iteration = train.load_checkpoint(
            checkpoint_path, model, optimizer)

    model.train()
    iteration = 0 # hardcoded irrelevant value
    train.validate(model, criterion, valset, iteration,
             hparams.batch_size, n_gpus, collate_fn, logger,
             hparams.distributed_run, rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=True, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("\n####\nRun Validation\n####\n")
    print("Checkpoint:", args.checkpoint_path)
    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    validate(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
