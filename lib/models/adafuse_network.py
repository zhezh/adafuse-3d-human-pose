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
import numpy as np
import random

from models.epipolar_fusion_layer import CamFusionModule, get_inv_cam, get_inv_affine_transform
from models.ada_weight_layer import ViewWeightNet
from models.soft_argmax import SoftArgmax2D
from core.loss import Joint2dSmoothLoss
from dataset import get_joint_mapping


np.seterr(divide='raise')


class MultiViewPose(nn.Module):
    def __init__(self, PoseResNet, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        general_joint_mapping = get_joint_mapping(self.config.DATASET.TRAIN_DATASET)
        reverse_joint_mapping = {general_joint_mapping[k]: k for k in range(20) if general_joint_mapping[k]!='*'}
        self.joint_mapping = []
        for k in sorted(reverse_joint_mapping.keys()):
            self.joint_mapping.append(reverse_joint_mapping[k])
        self.joint_mapping = torch.as_tensor(self.joint_mapping)

        self.resnet = PoseResNet
        self.b_crossview_fusion = self.config.CAM_FUSION.CROSSVIEW_FUSION
        self.b_ransac = True

        h = int(self.config.NETWORK.HEATMAP_SIZE[0])
        w = int(self.config.NETWORK.HEATMAP_SIZE[1])
        hm_sigma = self.config.NETWORK.SIGMA
        self.h = h
        self.w = w
        self.njoints = self.config.DATASET.NUM_USED_JOINTS
        self.nheatmaps = self.config.NETWORK.NUM_JOINTS
        self.nview = len(self.config.MULTI_CAMS.SELECTED_CAMS)
        self.cam_fusion_net = CamFusionModule(self.nview, self.njoints, self.h, self.w, general_joint_mapping, self.config)
        self.smax = SoftArgmax2D(window_fn='Uniform', window_width=5*hm_sigma, softmax_temp=0.05)
        self.view_weight_net = ViewWeightNet(self.config, self.nview)

        self.smooth2dloss = Joint2dSmoothLoss()

    def forward(self, inputs, **kwargs):
        dev = inputs.device
        # run_view_weight = kwargs['run_view_weight']
        run_phase = kwargs['run_phase']
        do_ransac = self.b_ransac and (run_phase == 'test')  # do not ransac when train, it is slow

        batch = inputs.shape[0]
        nview = inputs.shape[1]
        inputs = inputs.view(batch*nview, *inputs.shape[2:])
        njoints = len(self.joint_mapping)
        origin_hms, feature_before_final = self.resnet(inputs)
        # if not run_view_weight:  # todo, action not clear; does not need view weight
        #     return origin_hms, dict()

        # obtain camera in (batch, nview, ...)
        cam_R = kwargs['camera_R'].to(dev)  # (batch, nview, 3, 3)
        cam_T = kwargs['camera_T'].to(dev)
        cam_Intri = kwargs['camera_Intri'].to(dev)
        aug_trans = kwargs['aug_trans'].to(dev)  # origin full image -> heatmap
        inv_affine_trans = torch.inverse(aug_trans)  # heatmap -> origin image
        standard_cam_T = - torch.bmm(cam_R.view(-1,3,3), cam_T.view(-1,3,1)).view(batch, nview, 3, 1)
        # standard_cam_T: Translation same as in Hartley's book
        pmat = torch.bmm(self.collate_first_two_dims(cam_Intri),
                         self.collate_first_two_dims(torch.cat((cam_R, standard_cam_T), dim=3)))
        # # Notice: T is not h36m t, should be standard t
        pmat = pmat.view(batch, nview, 3, 4)
        fund_mats2 = self.get_fund_mat_pairs(cam_R, cam_T, cam_Intri)  # (batch, nview, nview, 3, 3)

        # camera in (batch*nview, ...)
        cam_R_collate = self.collate_first_two_dims(cam_R)
        cam_T_collate = self.collate_first_two_dims(cam_T)
        cam_Intri_collate = self.collate_first_two_dims(cam_Intri)
        aug_trans_collate = self.collate_first_two_dims(aug_trans)
        inv_affine_trans_collate = self.collate_first_two_dims(inv_affine_trans)

        # joint ground truth
        joint_vis = kwargs['joint_vis']  # ()
        gt_3d = kwargs['joints_gt'].to(dev)  # (batch, nview, njoints, 3)
        gt_3d = self.collate_first_two_dims(gt_3d)
        gt_3d = gt_3d.permute(0, 2, 1)  # (batch*nview, 3, njoints)
        gt_2d_cam = torch.bmm(cam_R_collate, (gt_3d - cam_T_collate))
        gt_2d = gt_2d_cam[:, 0:3] / gt_2d_cam[:, 2:]
        gt_2d = torch.bmm(cam_Intri_collate, gt_2d)
        gt_2d_hm = torch.bmm(aug_trans_collate, gt_2d)
        gt_2d_hm_row = gt_2d_hm.permute(0, 2, 1).contiguous()[:, :, 0:2]

        # 3d joint vis, especially for panoptic and totalcapture
        joints_vis_2d = kwargs['joints_vis']
        nviews_vis = torch.sum(joints_vis_2d, dim=1)[:, :, 0]
        nviews_vis = torch.index_select(nviews_vis, dim=1, index=self.joint_mapping.to(dev))
        nviews_vis = nviews_vis.view(batch, njoints, 1)

        # --- --- heatmap warp network
        xview_self_hm = self.cam_fusion_net(origin_hms, aug_trans_collate,
                                                                 cam_Intri_collate, cam_R_collate, cam_T_collate,
                                                                 inv_affine_trans_collate)
        xview_self_hm_added = torch.mean(xview_self_hm, dim=1)
        xview_self_hm_added = (xview_self_hm_added + origin_hms)/2.0  # crossview fusion with heuristic weights
        xview_self_hm_added = torch.index_select(xview_self_hm_added, dim=1, index=self.joint_mapping.to(dev))
        joint_heuristic, joint_heuristic_maxv = self.smax(xview_self_hm_added.detach())
        joint_heuristic_image = heatmap_coords_to_image(joint_heuristic, inv_affine_trans)
        joint_heuristic_image = joint_heuristic_image.view(batch, nview, 3, njoints)
        # --- End --- heatmap warp network

        # --- --- view weight network forward
        hms_nofusion = torch.index_select(origin_hms, dim=1, index=self.joint_mapping.to(dev))
        j2d_nofusion, j2d_nofusion_maxv = self.smax(hms_nofusion)
        j2d_nofusion_img = heatmap_coords_to_image(j2d_nofusion, inv_affine_trans)
        j2d_nofusion_img = j2d_nofusion_img.view(batch, nview, 3, njoints)
        confi = j2d_nofusion_maxv.view(batch, nview, njoints)
        distances = torch.zeros(batch, nview, nview-1, njoints).to(dev)
        confidences = torch.zeros(batch, nview, nview-1, njoints).to(dev)
        for b in range(batch):
            for i in range(nview):
                cv_joints = j2d_nofusion_img[b, i]  # cv-current view, col vector
                other_views = set(range(nview))
                other_views.remove(i)

                for idx_j, j in enumerate(other_views):
                    ov_joints = j2d_nofusion_img[b, j]  # ov-other view
                    # fund_mat = fund_mats[(b, j, i)]
                    fund_mat = fund_mats2[b,i,j]  # F_ij
                    l_i = torch.matmul(fund_mat, ov_joints)
                    distance_d = torch.sum(cv_joints * l_i, dim=0)**2
                    tmp_l_i = l_i**2
                    lp_i = torch.matmul(fund_mat.t(), cv_joints)
                    tmp_lp_i = lp_i**2
                    distance_div = tmp_l_i[0, :] + tmp_l_i[1, :] + tmp_lp_i[0, :] + tmp_lp_i[1, :]
                    distance = distance_d / distance_div  # Sampson first order here
                    distances[b, i, idx_j] = distance
                    confidences[b, i, idx_j] = confi[b, i]
        distances = torch.sqrt(distances)

        view_weight = self.view_weight_net(feat_map=feature_before_final.detach(), maxv=confi, heatmap=hms_nofusion.detach(),
                                           distances=distances, confidences=confidences)
        # view_weight (batch, njoint, nview)
        # --- End --- view weight network forward

        # --- fuse heatmaps with learned weight
        hms = hms_nofusion  # (batch*nview, n_used_joint, h, w)
        maxv = j2d_nofusion_maxv
        large_num = torch.ones_like(maxv) * 1e6
        maxv = torch.where(maxv>0.01, maxv, large_num)
        maxv = maxv.view(batch*nview, njoints, 1, 1)
        hms_norm = hms/maxv
        xview_self_hm_norm = self.cam_fusion_net(hms_norm, aug_trans_collate,
                                                                 cam_Intri_collate, cam_R_collate, cam_T_collate,
                                                                 inv_affine_trans_collate)
        warp_weight = self.get_warp_weight(view_weight)
        cat_hm = torch.cat((hms_norm.unsqueeze(dim=1), xview_self_hm_norm), dim=1)\
            .view(batch, nview, nview, njoints, self.h, self.w)
        fused_hm = warp_weight * cat_hm
        fused_hm = torch.sum(fused_hm, dim=2)
        fused_hm = fused_hm.view(batch * nview, *fused_hm.shape[2:])

        hms_out = torch.zeros_like(origin_hms)  # if we output hms_out, we need to convert its channel back to 20
        hms_out[:, self.joint_mapping] = fused_hm

        j2d_fused, j2d_fused_maxv, j2d_fused_smax = self.smax(fused_hm, out_smax=True)  # shape of (batch, 3, njoint) and (batch, njoint)
        j2d_fused_image = heatmap_coords_to_image(j2d_fused, inv_affine_trans)
        j2d_fused_image = j2d_fused_image.view(batch, nview, 3, njoints)
        j2d_fused_smax_out = torch.zeros_like(origin_hms)
        j2d_fused_smax_out[:, self.joint_mapping] = j2d_fused_smax
        # ---End- fuse heatmaps with learned weight

        # 2d joint loss here, should use smooth L1 loss
        joints_vis_3d = (nviews_vis >= 3).view(batch, 1, njoints).repeat(1, nview, 1)
        joints_vis_3d = joints_vis_3d.view(batch*nview, njoints, 1)
        joint_2d_non_homo = j2d_fused.permute(0,2,1)[:,:,0:2]
        smooth2dloss = self.smooth2dloss(joint_2d_non_homo, gt_2d_hm_row, target_weight=joints_vis_3d)

        #  --- do triangulation on multiple settings ---------------------------------------------------
        # no fusion
        j3d_nofusion = triangulation(j2d_nofusion_img, pmat, pts_mask=None)

        # ransac, takes a lot of time
        if do_ransac:
            j3d_ransac = ransac(pmat, j2d_nofusion_img, niter=10, epsilon=20)

        # use maxv as fusion weight applying on warp heatmap
        maxv_weight_fuse = j2d_nofusion_maxv.view(batch, nview, njoints).permute(0,2,1).contiguous()
        maxv_warp_weight = self.get_warp_weight(maxv_weight_fuse)
        cat_hm = torch.cat((hms_norm.unsqueeze(dim=1), xview_self_hm_norm), dim=1) \
            .view(batch, nview, nview, njoints, self.h, self.w)
        fused_hm = maxv_warp_weight * cat_hm
        fused_hm = torch.sum(fused_hm, dim=2)
        fused_hm = fused_hm.view(batch * nview, *fused_hm.shape[2:])
        j2d_maxv_fused, j2d_maxv_fused_maxv = self.smax(fused_hm)  # shape of (batch, 3, njoint) and (batch, njoint)
        j2d_maxv_fused_image = heatmap_coords_to_image(j2d_maxv_fused, inv_affine_trans)
        j2d_maxv_fused_image = j2d_maxv_fused_image.view(batch, nview, 3, njoints)
        j3d_maxv_fused = triangulation(j2d_maxv_fused_image, pmat, pts_mask=None)

        # heuristic fusion
        j3d_heuristic_fuse = triangulation(joint_heuristic_image, pmat, pts_mask=None)

        # weighted fusion
        mask_tmp = j2d_nofusion_maxv.view(batch, nview, njoints).permute(0,2,1).contiguous()
        mask_tmp2 = (mask_tmp > 0.01).type_as(mask_tmp)
        j2d_fused_image = j2d_fused_image.view(batch, nview, 3, njoints)
        j3d_ada_fused = triangulation(j2d_fused_image, pmat, pts_mask=mask_tmp2)

        out_extra = dict()
        out_extra['origin_hms'] = origin_hms
        out_extra['fused_hms_smax'] = j2d_fused_smax_out
        out_extra['joint_2d_loss'] = smooth2dloss
        out_extra['pred_view_weight'] = view_weight
        out_extra['maxv'] = maxv_weight_fuse  # (batch, njoints, nview)
        out_extra['nviews_vis'] = nviews_vis  # how many views are seen of grouping for each joint, obtained from gt
        out_extra['j3d_NoFuse'] = j3d_nofusion  # nofusion no weight, just triangulation
        out_extra['j3d_HeuristicFuse'] = j3d_heuristic_fuse  # crossview fusion with heuristic weight
        out_extra['j3d_ScoreFuse'] = j3d_maxv_fused
        if do_ransac:
            out_extra['j3d_ransac'] = j3d_ransac
        out_extra['j3d_AdaFuse'] = j3d_ada_fused

        out_extra['j2d_NoFuse'] = j2d_nofusion_img
        out_extra['j2d_HeuristicFuse'] = joint_heuristic_image
        out_extra['j2d_ScoreFuse'] = j2d_maxv_fused_image
        out_extra['j2d_AdaFuse'] = j2d_fused_image

        return hms_out, out_extra

    def get_warp_weight(self, view_weight):
        """

        :param view_weight: (batch, njoints, nview)
        :return: weights for merging warpped heatmap of shape (batch, nview, nview, njoints, 1, 1)
        """
        batch, njoints, nview = view_weight.shape
        dev = view_weight.device
        warp_weight = torch.zeros(batch, nview, nview, njoints).to(dev)
        warp_vw = view_weight.view(batch, njoints, nview).permute(0, 2, 1).contiguous()  # (batch, nview, njoint)
        for ci in range(nview):
            warp_weight[:, ci, 0] = warp_vw[:, ci]  # cur view weight at first
            # warp_weight[:, ci, 0] = 0
            all_views = list(range(nview))
            all_views.remove(ci)
            for idx, oi in enumerate(all_views):  # other views
                warp_weight[:, ci, idx + 1] = warp_vw[:, oi]
        warp_weight = warp_weight.view(*warp_weight.shape, 1, 1)
        return warp_weight

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def uncollate_first_two_dims(self, tensor, nbatch):
        """

        :param tensor: (batch*nview, ...)
        :param nbatch: int
        :return:
        """
        dim01 = tensor.shape[0]
        dim0 = nbatch
        dim1 = dim01//dim0
        left = tensor.shape[1:]
        return tensor.view(dim0, dim1, *left)

    def get_fund_mat_pairs(self, cam_R, cam_T, cam_Intri):
        """

        :param cam_R: (batch, nview, 3, 3)
        :param cam_T:
        :param cam_Intri:
        :return:
        """
        assert len(cam_R.shape) == 4, 'wrong shape of camera parameters'
        dev = cam_R.device
        batch, nview = cam_R.shape[0:2]
        # to get fundamental mat every two views, we need cam R,T,K in (batch, nview, nview-1)
        batch_camR_1 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camT_1 = torch.zeros(batch, nview, nview, 3, 1, device=dev)
        batch_camK_1 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camR_2 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camT_2 = torch.zeros(batch, nview, nview, 3, 1, device=dev)
        batch_camK_2 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        for b in range(batch):
            for i in range(nview):
                for j in range(nview):
                    batch_camR_1[b, i, j] = cam_R[b, j]
                    batch_camR_2[b, i, j] = cam_R[b, i]
                    batch_camT_1[b, i, j] = cam_T[b, j]
                    batch_camT_2[b, i, j] = cam_T[b, i]
                    batch_camK_1[b, i, j] = cam_Intri[b, j]
                    batch_camK_2[b, i, j] = cam_Intri[b, i]

        batch_camR_1 = batch_camR_1.view(-1, 3, 3)
        batch_camT_1 = batch_camT_1.view(-1, 3, 1)
        batch_camK_1 = batch_camK_1.view(-1, 3, 3)
        batch_camR_2 = batch_camR_2.view(-1, 3, 3)
        batch_camT_2 = batch_camT_2.view(-1, 3, 1)
        batch_camK_2 = batch_camK_2.view(-1, 3, 3)
        fund_mats2 = get_batch_fundamental_mat(batch_camR_1, batch_camT_1, batch_camK_1,
                                               batch_camR_2, batch_camT_2, batch_camK_2)
        fund_mats2 = fund_mats2.view(batch, nview, nview, 3, 3)
        return fund_mats2


def get_multiview_pose_net(resnet, CFG):
    model = MultiViewPose(resnet, CFG)
    return model


def get_fundamental_mat(r1, t1, k1, r2, t2, k2):
    """
    get fundamental mat, not in batch fashion
    l_2 = F cdot x_1;   x_2^T @ F @ x_1 = 0
    :param r1:
    :param t1:
    :param k1:
    :param r2:
    :param t2:
    :param k2:
    :return:
    """
    r = r2 @ r1.t()
    # t = -t1 + t2  # this t is general meaning, not special one in our h36m
    t = r2 @ (t1 - t2)  # t is h36m t.
    t_mat = torch.zeros(3,3).type_as(r1)  # cross product matrix
    t_mat[0,1] = -t[2]
    t_mat[0,2] = t[1]
    t_mat[1,2] = -t[0]
    t_mat = -t_mat.t() +t_mat
    fundmat = torch.inverse(k2).t() @ t_mat @ r @ torch.inverse(k1)
    return fundmat


def get_batch_fundamental_mat(r1, t1, k1, r2, t2, k2):
    """

    :param r1:
    :param t1: in h36m t style
    :param k1:
    :param r2:
    :param t2:
    :param k2:
    :return:
    """
    nbatch = r1.shape[0]
    r = torch.bmm(r2, r1.permute(0,2,1))
    t = torch.bmm(r2, (t1 - t2))  # t is h36m t.
    t = t.view(nbatch,3)
    t_mat = torch.zeros(nbatch, 3, 3).type_as(r1)  # cross product matrix
    t_mat[:, 0, 1] = -t[:, 2]
    t_mat[:, 0, 2] = t[:, 1]
    t_mat[:, 1, 2] = -t[:, 0]
    t_mat = -t_mat.permute(0,2,1) + t_mat
    inv_k1 = torch.inverse(k1)
    inv_k2 = torch.inverse(k2)
    inv_k2_t = inv_k2.permute(0,2,1)
    fundmat = torch.bmm(inv_k2_t, torch.bmm(t_mat, torch.bmm(r, inv_k1)))

    return fundmat


def triangulation(pts, pmat, distor=None, pts_mask=None):
    """

    :param pts: (batch, nview, 3, njoints)
    :param pmat: camera projection matrix of shape (batch, nview,3,4), pmat = K*[R|t], note t is different from t in out h36m definition
    :param distor:
    :param pts_mask: indicating which joints involve in triangulation, should be shape of (batch, njoint, nview)
    :return:
    """
    with torch.no_grad():
        dev = pts.device
        pts = pts.permute(0,3,1,2).contiguous().to(dev)  # (batch, njoints, nview, 3)
        batch, njoint, nview = pts.shape[0:3]
        pmat = pmat.to(dev)

        if pts_mask is not None:
            pts_mask = pts_mask.to(dev)
            pts_mask = pts_mask.view(batch * njoint, nview, 1)
            pts_mask = pts_mask.expand(-1, -1, 2).contiguous().view(batch * njoint, 2 * nview, 1)
            # (batch*njoint, 2nview, 1)
            view_weights = pts_mask
        else:
            view_weights = torch.ones(batch*njoint, 2*nview, 1).to(dev)

        # reshape pts to (batch*njoint, nview, 3), and get A as (batch*njoint, 2*nview, 4)
        A = torch.zeros(batch*njoint, 2 * nview, 4).to(dev)
        pts_compact = pts.view(batch*njoint, nview, 3)
        for i in range(nview):
            Pmat = pmat[:, i]  # (batch, 3, 4)
            Pmat = Pmat.view(batch, 1, 3, 4).expand(-1, njoint, -1, -1).contiguous().view(batch*njoint, 3, 4)
            row0 = Pmat[:, 0]  # (batch, 4)
            row1 = Pmat[:, 1]
            row2 = Pmat[:, 2]
            x = pts_compact[:, i, 0:1]  # (batch, 1)
            y = pts_compact[:, i, 1:2]  # (batch, 1)

            A[:, 2*i] = (x*row2 - row0)
            A[:, 2*i+1] = y*row2 - row1

        A = view_weights * A
        A_np = A.cpu().numpy()
        try:
            u, d, vt = np.linalg.svd(A_np)  # vt (batch*njoint, 4, 4)
            Xs = vt[:,-1,0:3]/vt[:,-1,3:]
        except np.linalg.LinAlgError:
            Xs = np.zeros((batch*njoint, 3), dtype=np.float32)
        except FloatingPointError:
            # print(vt[:,-1,3:])
            div = vt[:,-1,3:]
            div[div==0] = float('inf')
            Xs = vt[:,-1,0:3]/vt[:,-1,3:]

        # convert ndarr to tensor
        Xs = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
        Xs = Xs.view(batch, njoint, 3)
        return Xs


def ransac(pmat, points, niter=10, epsilon=15):
    """

    :param pmat: (batch, nview, 3, 4)
    :param points: (batch, nview, 3, njoints)
    :param niter:
    :return:
    """
    assert pmat.shape[1] == points.shape[1]
    batch = pmat.shape[0]
    nview = pmat.shape[1]
    njoint = points.shape[3]

    out_kp3ds = torch.zeros(batch, njoint, 3).to(pmat.device)

    for b in range(batch):
        for j in range(njoint):

            pmat_ = pmat[b:b+1]
            points_ = points[b:b+1, :, :, j:j+1]  # (1, nview, 3, 1)

            view_set = set(range(nview))
            inlier_set = set()
            for i in range(niter):
                sampled_views = sorted(random.sample(view_set, 2))
                pts_mask = torch.zeros(1, 1, nview)
                pts_mask[:,:,sampled_views] = 1
                kp3d = triangulation(pts=points_, pmat=pmat_, pts_mask=pts_mask)  # (1, 1, 3)
                kp3d = kp3d.cpu().numpy()

                # reprojection error
                kp3d = np.reshape(kp3d, (1,3))  # (1, 3, 1)
                ones = np.ones((1, 1))
                kp4d = np.concatenate([kp3d, ones], axis=1)  # (1, 4)
                kp4d = kp4d.reshape(-1)
                pmat_2 = pmat[b].cpu().numpy()  # (nview, 3, 4)

                kp2d = np.matmul(pmat_2, kp4d)  # (nview, 3, 1)
                kp2d = kp2d.reshape((nview, 3))
                kp2d = kp2d / kp2d[:, 2:3]
                points_2 = points[b, :, :, j].cpu().numpy()
                reprojection_error = np.sqrt(np.sum((kp2d - points_2)**2, axis=1))

                new_inlier_set = set([i for i, v in enumerate(reprojection_error) if v < epsilon])

                if len(new_inlier_set) > len(inlier_set):
                    inlier_set = new_inlier_set
                if len(inlier_set) == nview:
                    break
            if len(inlier_set) < 2:
                inlier_set = view_set.copy()

            pts_mask = torch.zeros(1, 1, nview)
            pts_mask[:,:,sorted(list(inlier_set))] = 1
            kp3d = triangulation(pts=points_, pmat=pmat_, pts_mask=pts_mask)  # (1, 1, 3)
            out_kp3ds[b, j] = kp3d[0,0]

    return out_kp3ds


def heatmap_coords_to_image(coords, inv_affine_trans):
    """

    :param coords: (batch*nview, 3, njoints)
    :param inv_affine_trans: (batch, nview, 3, 3)
    :return:
    """
    if len(inv_affine_trans.shape) == 4:
        inv_affine_trans = inv_affine_trans.view(-1, 3, 3)
    coords_img = torch.bmm(inv_affine_trans, coords)
    return coords_img
