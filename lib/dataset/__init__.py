# ------------------------------------------------------------------------------
# pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.mpii import MPIIDataset as mpii
from dataset.multiview_h36m import MultiViewH36M as multiview_h36m
from dataset.mixed_dataset import MixedDataset as mixed
from dataset.multiview_mpii import MultiviewMPIIDataset as multiview_mpii
from dataset.unrealcv_dataset import UnrealcvData as unrealcv


# joint mapping
def get_joint_mapping(dataset_name):
    if dataset_name in ['multiview_h36m']:
        general_joint_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: '*', 9: 8, 10: '*', 11: 9, 12: 10,
                                 13: '*', 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16}
    elif dataset_name == 'totalcapture':
        general_joint_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: '*', 9: 8, 10: '*', 11: 9,
                                 12: '*',
                                 13: '*', 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15}
    elif dataset_name == 'panoptic':
        general_joint_mapping = {0: 2, 1: 12, 2: 13, 3: 14, 4: 6, 5: 7, 6: 8, 7: 15, 8: 16, 9: 0, 10: '*', 11: 1,
                                 12: 17, 13: 18, 14: 3, 15: 4, 16: 5, 17: 9, 18: 10, 19: 11}
    elif dataset_name in ['unrealcv']:
        general_joint_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: '*', 9: 8, 10: '*', 11: '*',
                                 12: '*',
                                 13: '*', 14: 9, 15: 10, 16: 11, 17: 12, 18: 13, 19: 14}
    else:
        assert 0 == 1, 'Not defined dataset'

    return general_joint_mapping
