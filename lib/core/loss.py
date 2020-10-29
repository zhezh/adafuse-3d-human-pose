# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss


class JointMPJPELoss(nn.Module):
    def __init__(self):
        super(JointMPJPELoss, self).__init__()

    def forward(self, joint_3d, gt, joints_vis_3d=None, output_batch_mpjpe=False):
        """

        :param joint_3d: (batch, njoint, 3)
        :param gt:
        :param joints_vis_3d: (batch, njoint, 1), values are 0,1
        :param output_batch_mpjpe: bool
        :return:
        """
        if joints_vis_3d is None:
            joints_vis_3d = torch.ones_like(joint_3d)[:,:,0:1]
        l2_distance = torch.sqrt(((joint_3d - gt)**2).sum(dim=2))
        joints_vis_3d = joints_vis_3d.view(*l2_distance.shape)
        masked_l2_distance = l2_distance * joints_vis_3d
        n_valid_joints = torch.sum(joints_vis_3d, dim=1)
        # if (n_valid_joints < 1).sum() > 0:
        n_valid_joints[n_valid_joints < 1] = 1  # avoid div 0
        avg_mpjpe = torch.sum(masked_l2_distance) / n_valid_joints.sum()
        if output_batch_mpjpe:
            return avg_mpjpe, masked_l2_distance, n_valid_joints.sum()
        else:
            return avg_mpjpe, n_valid_joints.sum()


class Joint2dSmoothLoss(nn.Module):
    def __init__(self):
        super(Joint2dSmoothLoss, self).__init__()
        factor = torch.as_tensor(8.0)
        alpha = torch.as_tensor(-10.0)
        self.register_buffer('factor', factor)
        self.register_buffer('alpha', alpha)

    def forward(self, joint_2d, gt, target_weight=None):
        """

        :param joint_2d: (batch*nview, njoint, 2)
        :param gt:
        :param target_weight: (batch*nview, njoint, 1)
        :return:
        """
        x = torch.sum(torch.abs(joint_2d - gt), dim=2)  # (batch*nview, njoint)
        x_scaled = ((x / self.factor) ** 2 / torch.abs(self.alpha-2) + 1) ** (self.alpha * 0.5) -1
        x_final = (torch.abs(self.alpha) - 2) / self.alpha * x_scaled

        loss = x_final
        if target_weight is not None:
            cond = torch.squeeze(target_weight) < 0.5
            loss = torch.where(cond, torch.zeros_like(loss), loss)
        loss_mean = loss.mean()
        return loss_mean * 1000.0
