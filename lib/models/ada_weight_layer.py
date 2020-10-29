# ------------------------------------------------------------------------------
# adafuse
# Copyright (c) 2019-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zhe Zhang (v-zhaz@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewWeightNet(nn.Module):
    def __init__(self, config, nview):
        super().__init__()
        self.config = config
        self.njoints = self.config.DATASET['NUM_USED_JOINTS']
        self.h = int(self.config.NETWORK.HEATMAP_SIZE[0])
        self.w = int(self.config.NETWORK.HEATMAP_SIZE[1])
        self.nview = nview
        BN_MOMENTUM = 0.1

        self.nchan_per_joint = 128  # feature vector for each joint
        self.nchan_dist_confi = 256
        self.nchan_vw = 128

        self.joint_feat_channel = 1  # use heatmap instead
        self.heatmap_feature_net = nn.Sequential(
            nn.Conv2d(self.joint_feat_channel, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        self.joint_feature_net = nn.Sequential(
            nn.Conv2d(256, self.nchan_per_joint, 1),
            nn.BatchNorm2d(self.nchan_per_joint, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.dist_feature_net = nn.Sequential(
            nn.Conv1d(2, self.nchan_dist_confi, 1),
            nn.BatchNorm1d(self.nchan_dist_confi),
            nn.ReLU(inplace=True),

            nn.Conv1d(self.nchan_dist_confi, self.nchan_dist_confi, 1),
            nn.BatchNorm1d(self.nchan_dist_confi),
            nn.ReLU(inplace=True),
        )

        n_out_fc = self.nchan_per_joint + self.nchan_dist_confi
        self.conf_out = nn.Sequential(
            nn.Linear(n_out_fc, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_map, maxv, heatmap, distances, confidences):
        """

        :param feat_map: (batch*nview, 256, h, w)
        :param maxv: (batch*nview, njoints)
        :param heatmap: (batch*nview, n_used_joint, h, w)
        :param distances: (batch, nview, nview-1, njoints), unit in px
        :param confidences: (batch, nview, nview-1, njoints)
        :return:
        """
        dev = feat_map.device
        batch = feat_map.shape[0] // self.nview
        nview = self.nview
        njoints = self.njoints

        heatmap = heatmap.view(batch*nview*njoints, 1, self.h, self.w)
        heatmap_feats = self.heatmap_feature_net(heatmap)  # (batch*nview*njoint, 256, h/4, w/4)
        heatmap_feats_avg = F.adaptive_avg_pool2d(heatmap_feats, 1)  # (batch*nview*njoint, 256, 1, 1)
        heatmap_feats = self.joint_feature_net(heatmap_feats_avg)  # (batch*nview*njoint, 256, 1, 1)
        heatmap_feats = heatmap_feats.view(batch, self.nview, self.njoints, self.nchan_per_joint).permute(0,2,1,3).contiguous()
        joint_features = heatmap_feats.view(batch * self.njoints * self.nview, -1)

        # distances (batch, nview, nview-1, njoint)
        # distances = 1/(distances+1e-3)
        distances = torch.exp(-distances)
        distances = distances.permute(0, 3, 1, 2).contiguous().view(batch*njoints*nview, 1, nview-1)
        confidences = confidences.permute(0, 3, 1, 2).contiguous().view(batch*njoints*nview, 1, nview-1)
        feat = torch.cat((distances, confidences), dim=1).contiguous()
        feat = self.dist_feature_net(feat)
        dist_feat = torch.mean(feat, dim=2, keepdim=True)
        dist_feat = dist_feat.view(batch * self.njoints * self.nview, -1)

        final_feature = torch.cat((joint_features, dist_feat), dim=1)
        logit = self.conf_out(final_feature)
        x = logit.view(batch, njoints, nview)

        x_zeros = torch.zeros_like(x)
        maxv_tmp = maxv.view(batch, nview, njoints).permute(0,2,1).contiguous()
        x = torch.where(maxv_tmp>0.01, x, x_zeros)

        return x
