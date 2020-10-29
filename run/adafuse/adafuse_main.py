from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
from pathlib import Path

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from git import Repo

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss, JointMPJPELoss
from core.adafuse_function import run_model
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger
import dataset
import models

from dataset.adafuse_collate import adafuse_collate


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        required=True,
        type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument(
        '--workers',
        help='num of dataloader workers',
        type=int)
    parser.add_argument(
        '--cams', help='view/cam id to use', type=str)
    # parser.add_argument(
    #     '--ablation', help='heatmap_only, consistency_only, both', type=str)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    parser.add_argument(
        '--runMode', help='train or test', type=str, default='train')
    parser.add_argument(
        '--modelFile', help='model for test', type=str, default='0')
    parser.add_argument(
        '--evaluate', help='directly use provided model to evaluate results', type=str2bool, default='0')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.cams:
        selected_cams = [int(c) for c in args.cams.split(',')]
        config.MULTI_CAMS.SELECTED_CAMS = selected_cams


def main():
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=300)

    args = parse_args()
    reset_config(config, args)
    run_phase = args.runMode  # train or test

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, run_phase)

    model_file = 'final_state_ep{}.pth.tar'.format(args.modelFile)
    # print code version info
    try:
        repo = Repo('')
        repo_git = repo.git
        working_tree_diff_head = repo_git.diff('HEAD')
        this_commit_hash = repo.commit()
        cur_branches = repo_git.branch('--list')
        logger.info('Current Code Version is {}'.format(this_commit_hash))
        logger.info('Current Branch Info :\n{}'.format(cur_branches))
        logger.info('Working Tree diff with HEAD: \n{}'.format(
            working_tree_diff_head))
    except:
        logger.info('Git repo not initialized')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=True)
    model = models.adafuse_network.get_multiview_pose_net(
        backbone_model, config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # load pretrained backbone
    # Note this backbone is already trained on current dataset
    pretrained_backbone_file = Path(config.DATA_DIR) / config.NETWORK.PRETRAINED
    if os.path.exists(pretrained_backbone_file):
        model.load_state_dict(torch.load(pretrained_backbone_file), strict=False)

    if args.evaluate:
        run_phase = 'test'
        model_file_path = config.NETWORK.ADAFUSE
        model.load_state_dict(torch.load(model_file_path), strict=True)
        logger.info('=> loading model from {} for evaluating'.format(model_file_path))
    elif run_phase == 'test':
        model_state_file = os.path.join(final_output_dir, model_file)
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file), strict=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    criterion_mpjpe = JointMPJPELoss().cuda()

    view_weight_params = []
    for name, param in model.named_parameters():
        if 'view_weight_net' in name:
            param.requires_grad = True
            view_weight_params.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(params=view_weight_params, lr=config.TRAIN.LR)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    if run_phase == 'train' and config.TRAIN.RESUME:
        start_epoch, model, optimizer, ckpt_perf = load_checkpoint(
            model, optimizer, final_output_dir)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if run_phase == 'train':
        train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
            config, config.DATASET.TRAIN_SUBSET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            collate_fn=adafuse_collate,
            pin_memory=True)

    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        collate_fn=adafuse_collate,
        pin_memory=True)

    if run_phase == 'train':
        best_perf = ckpt_perf
        best_epoch = -1
        best_model = False
    perf_indicator = 0
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        extra_param = dict()
        extra_param['loss_mpjpe'] = criterion_mpjpe

        if run_phase == 'train':
            params = {'config': config,
                      'dataset': train_dataset,
                      'loader': train_loader,
                      'model': model,
                      'criterion_mse': criterion,
                      'criterion_mpjpe': criterion_mpjpe,
                      'final_output_dir': final_output_dir,
                      'tb_writer': writer_dict,
                      'optimizer': optimizer,
                      'epoch': epoch,
                      'is_train': True,
                      'save_heatmaps': False,}
            # train
            run_model(**params)

            # save checkpoint and model before validation
            if divmod(epoch + 1, 1)[1] == 0:  # save checkpoint every x epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': get_model_name(config),
                    'state_dict': model.module.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, False, final_output_dir, filename='checkpoint_ep{}.pth.tar'.format(epoch))

            # save final state at every epoch
            final_model_state_file = os.path.join(
                final_output_dir, 'final_state_ep{}.pth.tar'.format(epoch))
            logger.info('saving final model state to {}'.format(
                final_model_state_file))
            torch.save(model.module.state_dict(), final_model_state_file)

        valid_params = {'config': config,
                        'dataset': valid_dataset,
                        'loader': valid_loader,
                        'model': model,
                        'criterion_mse': criterion,
                        'criterion_mpjpe': criterion_mpjpe,
                        'final_output_dir': final_output_dir,
                        'tb_writer': writer_dict,
                        'optimizer': optimizer,
                        'epoch': epoch,
                        'is_train': False,
                        'save_heatmaps': False, }
        perf_indicator = run_model(**valid_params)

        if run_phase == 'test':
            break  # if run mode is test, only run test one time is enough

        logger.info(
            '=> perf indicator at epoch {} is {}. old best is {} '.format(
                epoch, perf_indicator, best_perf))

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
            best_epoch = epoch
            logger.info(
                '====> find new best model at end of epoch {}. (start from 0)'.format(epoch))
        else:
            best_model = False
        logger.info(
            'epoch of best validation results is {}'.format(best_epoch))

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
    # --- End all epoch
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
