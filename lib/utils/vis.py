# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
import os
import matplotlib.pyplot as plt

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 file_name,
                                 nrow=8,
                                 padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                               [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, show_pred_dot=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)*4
    heatmap_width = batch_heatmaps.size(3)*4

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0])*4, int(preds[i][j][1])*4), 1,
                       [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            colored_heatmap = cv2.resize(colored_heatmap,
                                   (int(heatmap_width), int(heatmap_height)))
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            if show_pred_dot:
                cv2.circle(masked_image, (int(preds[i][j][0])*4, int(preds[i][j][1])*4),
                           1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_fusion_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, out_large=True):
    """
    For orient 2d fused heatmap  -- By Zhe Zhang
    :param batch_image:
    :param batch_heatmaps:
    :param file_name:
    :param normalize:
    :return:
    """
    batch_fusion_heatmaps_min = torch.min(batch_heatmaps)
    batch_fusion_heatmaps_max = torch.max(batch_heatmaps)
    batch_fusion_heatmaps = (batch_heatmaps - batch_fusion_heatmaps_min) / (
                batch_fusion_heatmaps_max - batch_fusion_heatmaps_min + 1e-9)
    if out_large:
        save_batch_heatmaps_large(batch_image, batch_fusion_heatmaps, file_name, normalize)
    else:
        save_batch_heatmaps(batch_image, batch_fusion_heatmaps, file_name, normalize)


def save_batch_heatmaps_large(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_heatmaps = F.interpolate(batch_heatmaps, size=batch_image.shape[2:4])

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])), 1,
                       [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(input, meta['joints_2d_transformed'],
                                     meta['joints_vis'],
                                     '{}_gt.jpg'.format(prefix1))
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(input, joints_pred, meta['joints_vis'],
                                     '{}_pred.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(input, target, '{}_hm_gt.jpg'.format(prefix2))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(input, output, '{}_hm_pred.jpg'.format(prefix2))


def save_debug_fused_images(config, input, meta, target, joints_pred, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(input, joints_pred, meta['joints_vis'],
                                     '{}_pred_fuse.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(input, output, '{}_hm_pred_fuse.jpg'.format(prefix2))


def save_debug_images_2(config, input, meta, target, joints_pred, output, prefix, suffix='fuse', normalize=False, IMG=True, HMS=True):
    """

    :param config:
    :param input: input image
    :param meta:
    :param target:
    :param joints_pred:
    :param output: heatmap
    :param prefix:
    :param suffix: appear in image file name
    :param normalize: normalize heatmap to [0,1]
    :param IMG: if saving debug joint image
    :param HMS: if saving debug joint heatmap
    :return:
    """
    if not config.DEBUG.DEBUG:
        return

    if normalize:
        nimg, njoints = output.shape[0:2]
        batch_fusion_heatmaps_min, _ = torch.min(output.view(nimg, njoints, -1), dim=2)
        batch_fusion_heatmaps_max, _ = torch.max(output.view(nimg, njoints, -1), dim=2)
        batch_fusion_heatmaps_min = batch_fusion_heatmaps_min.view(nimg, njoints, 1, 1)
        batch_fusion_heatmaps_max = batch_fusion_heatmaps_max.view(nimg, njoints, 1, 1)
        batch_fusion_heatmaps = (output - batch_fusion_heatmaps_min) / (
                batch_fusion_heatmaps_max - batch_fusion_heatmaps_min + 1e-6)
        output = batch_fusion_heatmaps

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if IMG:
        save_batch_image_with_joints(input, joints_pred, meta['joints_vis'],
                                     '{}_pred_{}.jpg'.format(prefix1, suffix))
    if HMS:
        save_batch_heatmaps(input, output, '{}_hm_pred_{}.jpg'.format(prefix2, suffix), show_pred_dot=False)
