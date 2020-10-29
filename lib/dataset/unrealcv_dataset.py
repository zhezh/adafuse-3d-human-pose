from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections
from operator import itemgetter
import logging

from dataset.joints_dataset import JointsDataset
import multiviews.cameras as cam_utils


logger = logging.getLogger()


class UnrealcvData(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'lsho',
            10: 'lelb',
            11: 'lwri',
            12: 'rsho',
            13: 'relb',
            14: 'rwri'
        }

        self.u2a_mapping = super().get_mapping()
        anno_file = osp.join(self.root, 'unrealcv', 'annot',
                             'unrealcv_{}.pkl'.format(image_set))
        self.db = self.load_db(anno_file)
        for datum in self.db:
            joints_vis_3d = datum['joints_vis']
            joints_vis_2d = datum['joints_vis_2d']
            joints_vis_new = np.concatenate([joints_vis_2d]*3, axis=1)
            datum['joints_vis_3d'] = joints_vis_3d
            datum['joints_vis'] = joints_vis_new

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()
        self.grouping = self.get_group(self.db)
        self.selected_cam = list(cfg.MULTI_CAMS.SELECTED_CAMS)  # borrow from totalcapture
        self.grouping = self.filter_views_from_group()  # filter out certain views

        if self.is_train:
            self.grouping = self.grouping[::5]
        else:
            self.grouping = self.grouping[::5]

        self.group_size = len(self.grouping)

    def index_to_action_names(self):
        return {
            0: 'basketball',
            1: 'dance',
            2: 'dance2',
            3: 'martial',
            4: 'soccer',
            5: 'boxing',
            6: 'exercise',
        }

    def filter_views_from_group(self):
        filtered_group = []
        for g in self.grouping:
            ng = itemgetter(*self.selected_cam)(g)
            filtered_group.append(ng)
        return filtered_group

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1, -1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item, source='unrealcv')
            # data type convert to float32
            m['scale'] = m['scale'].astype(np.float32)
            m['center'] = m['center'].astype(np.float32)
            m['rotation'] = int(m['rotation'])
            if 'name' in m['camera']:
                del m['camera']['name']
            for k in m['camera']:
                m['camera'][k] = m['camera'][k].astype(np.float32)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return '{}_{:06}'.format(datum['video_id'][:-3], datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()
        nview = len(self.selected_cam)
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.0125

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        flat_items = []
        box_lengthes = []
        vis_3d = []
        vis_2d = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
                # vis_3d.append(self.db[item]['joints_vis'][su, :1])  # it is been changed when do mapping
                # in fact, we use joints_vis_2d to generate joints_vis
                # the indicator of whether joint in img view, is joints_vis_3d
                vis_2d.append(self.db[item]['joints_vis_2d'])
                vis_3d.append(self.db[item]['joints_vis_3d'])
                flat_items.append(self.db[item])
                boxsize = np.array(self.db[item]['scale']).sum() * 100.0
                box_lengthes.append(boxsize)
        gt = np.array(gt)
        vis_2d = np.squeeze(np.array((vis_2d)).astype(np.bool), axis=2)
        vis_3d = np.squeeze(np.array((vis_3d)).astype(np.bool), axis=2)

        if pred.shape[1] == 20:
            pred = pred[:, su, :2]
        elif pred.shape[1] == 15:
            pred = pred[:, :, :2]

        detection_threshold = np.array(box_lengthes).reshape((-1, 1)) * threshold

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= detection_threshold)

        # detection rate of those in the img view
        inview_joints_each_type = np.sum(vis_3d, axis=0)
        joint_detection_rate = np.sum(detected, axis=0) / inview_joints_each_type

        # unrealcv specific statistics
        # 2d occluded, 3d vis, detection rate
        selected_joints = np.logical_and(vis_3d, np.logical_not(vis_2d))
        num_occluded_each_type = np.sum(selected_joints, axis=0)
        occluded_but_detected = np.logical_and(selected_joints, detected)
        occluded_detection_rate = np.sum(occluded_but_detected, axis=0) / num_occluded_each_type

        for i in range(len(a2u)):
            logger.info('type {} has {:8d} occluded joints, detection rate {:.6f}'
                        .format(self.actual_joints[sa[i]],
                                num_occluded_each_type[i],
                                occluded_detection_rate[i]))
        logger.info('avg occluded detection rate for threshold {} is: {}'
                    .format(threshold,
                            np.sum(occluded_but_detected) / num_occluded_each_type.sum()))

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]

        for i in range(len(a2u)):
            name_values['{}D'.format(joint_names[sa[i]])] = occluded_detection_rate[i]

        detected_int = detected.astype(np.int)
        nsamples, njoints = detected.shape
        per_grouping_detected = detected_int.reshape(nsamples // nview, nview * njoints)
        return name_values, np.mean(joint_detection_rate), per_grouping_detected

    def evaluate_3d(self, preds3d, thresholds=None):
        if thresholds is None:
            thresholds = [5., 10., 15., 20., 25., 50., 75., 100., 125., 150.,]

        gt3d = []
        for idx, items in enumerate(self.grouping):
            # note that h36m joints_3d is in camera frame
            db_rec = self.db[items[0]]
            j3d_global = cam_utils.camera_to_world_frame(db_rec['joints_3d'], db_rec['camera']['R'], db_rec['camera']['T'])
            gt3d.append(j3d_global)
        gt3d = np.array(gt3d)

        assert preds3d.shape == gt3d.shape, 'shape mismatch of preds and gt'
        distance = np.sum((preds3d - gt3d)**2, axis=2)

        num_groupings = len(gt3d)
        pcks = []
        for thr in thresholds:
            detections = distance <= thr**2
            detections_perjoint = np.sum(detections, axis=0)
            pck_perjoint = detections_perjoint / num_groupings
            # pck_avg = np.average(pck_perjoint, axis=0)
            pcks.append(pck_perjoint)

        return thresholds, pcks
