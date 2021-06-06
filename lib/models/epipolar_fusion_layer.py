from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


tv1, tv2, _ = torch.__version__.split('.')
tv = int(tv1) * 10 + int(tv2) * 1
if tv >= 13:  # api change since 1.3.0 for grid_sample
    grid_sample = functools.partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample


def gen_hm_grid_coords(w, h, dev=None):
    """

    :param h:
    :param w:
    :param dev:
    :return: (3, h*w) each col is (u, v, 1)^T
    """
    if not dev:
        dev = torch.device('cpu')
    h = int(h)
    w = int(w)
    h_s = torch.linspace(0, h - 1, h).to(dev)
    w_s = torch.linspace(0, w - 1, w).to(dev)
    hm_cords = torch.meshgrid(h_s, w_s)
    flat_cords = torch.stack(hm_cords, dim=0).view(2, -1)
    out_grid = torch.ones(3, h*w, device=dev)
    out_grid[0] = flat_cords[1]
    out_grid[1] = flat_cords[0]
    return out_grid


def get_inv_cam(intri_mat, extri_R, extri_T):
    """
    all should be in (nview*batch, x, x)
    :param intri_mat:
    :param extri_R:
    :param extri_T:
    :return:
    """
    # camera_to_world  torch.mm(torch.t(R), x) + T
    # world_to_camera  torch.mm(R, x - T)
    # be aware of that: extri T is be add and minus in cam->world and reverse
    return torch.inverse(intri_mat), extri_R.permute(0,2,1).contiguous(), extri_T


def get_inv_affine_transform(affine_t):
    """

    :param affine_t: (3x3) mat instead of 2x3 mat. shape of (nview*batch, 3, 3)
    :return:
    """
    return torch.inverse(affine_t)


class CamFusionModule(nn.Module):
    def __init__(self, nview, njoint, h, w, joint_hm_mapping, config):
        super().__init__()
        self.nview = nview
        # self.batch = batch
        self.njoint = njoint  # njoint in heatmap, normally 20
        self.h = h  # h of heatmap
        self.w = w  # w of heatmap
        self.joint_hm_mapping = joint_hm_mapping
        self.config = config
        self.b_crossview_fusion = config.CAM_FUSION.CROSSVIEW_FUSION
        self.onehm = gen_hm_grid_coords(h, w)

    def forward(self, heatmaps, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans):
        dev = heatmaps.device
        batch = heatmaps.shape[0] // self.nview
        self.onehm = self.onehm.to(dev)

        cam_Intri = cam_Intri.to(dev)
        cam_R = cam_R.to(dev)
        cam_T = cam_T.to(dev)
        affine_trans = affine_trans.to(dev)
        inv_affine_trans = inv_affine_trans.to(dev)
        inv_cam_Intri, inv_cam_R, inv_cam_T = get_inv_cam(cam_Intri, cam_R, cam_T)

        crossview_fused = None
        if self.b_crossview_fusion:
            xview_self_depth = [1000., 5000.]
            # uv1 to global with 2 depth
            nc, hw = self.onehm.shape
            uvhm_coords = self.onehm.view(1, nc, hw).expand(self.nview * batch, -1, -1)
            uvhm_coords = uvhm_coords.contiguous()
            xs_global = []
            for dep in xview_self_depth:
                dep_mul_factor = torch.tensor([[dep]], device=dev)
                x_g = torch.bmm(inv_cam_R, dep_mul_factor * torch.bmm(inv_cam_Intri, torch.bmm(inv_affine_trans, uvhm_coords))) + inv_cam_T
                xs_global.append(x_g)

            coords_flow = torch.zeros(batch, self.nview, self.nview-1, self.h*self.w, self.h+self.w, 3, device=dev)
            # project to a view
            all_views = set(range(self.nview))
            for curview in range(self.nview):
                other_views = all_views.copy()
                other_views.remove(curview)
                for idx_othview, othview in enumerate(other_views):
                    ref_pts = []
                    for idxpt in range(2):
                        tmp_x_g = xs_global[idxpt][curview::self.nview]

                        tmp_R = cam_R[othview::self.nview]
                        tmp_T = cam_T[othview::self.nview]
                        tmp_K = cam_Intri[othview::self.nview]
                        tmp_Aff = affine_trans[othview::self.nview]

                        xcam = torch.bmm(tmp_R, (tmp_x_g-tmp_T))
                        xnormim = xcam/xcam[:,2:3]
                        x_uv_hm = torch.bmm(tmp_Aff, torch.bmm(tmp_K, xnormim))
                        ref_pts.append(x_uv_hm)
                        # print(curview, othview, idxpt)
                    # get flow for grid_sample
                    # k = (y1-y0)/(x1-x0)
                    ref_pt0 = ref_pts[0]  # (batch, 3, h*w)
                    ref_pt1 = ref_pts[1]

                    kk = ((ref_pt1[:, 1] - ref_pt0[:, 1]) / (ref_pt1[:, 0] - ref_pt0[:, 0])).view(batch, 1, -1)
                    xs = torch.tensor(list(range(self.w)), dtype=torch.float32, device=dev).view(1,-1,1)\
                        .expand(batch,-1,self.h*self.w)  # (batch, w, 1)
                    x0 = ref_pt0[:,0].view(batch,1,-1).expand(-1,self.w,-1)
                    y0 = ref_pt0[:,1].view(batch,1,-1).expand(-1,self.w,-1)
                    ys = kk * (xs - x0) + y0
                    coords_flow[:, curview, idx_othview, :, :self.w, 0] = xs.permute(0,2,1)
                    coords_flow[:, curview, idx_othview, :, :self.w, 1] = ys.permute(0,2,1)

                    ys = torch.tensor(list(range(self.h)), dtype=torch.float32, device=dev).view(1, -1, 1) \
                        .expand(batch, -1, self.h * self.w)
                    xs = (ys - y0) / kk + x0
                    coords_flow[:, curview, idx_othview, :, self.w:, 0] = xs.permute(0, 2, 1)
                    coords_flow[:, curview, idx_othview, :, self.w:, 1] = ys.permute(0, 2, 1)

                    coords_flow[:, curview, idx_othview, :, :, 2] = othview

            # grid smaple
            flow_norm_factor = torch.tensor([self.w-1, self.h-1, self.nview-1], dtype=torch.float32, device=dev)/2
            grid = coords_flow.view(batch, self.nview*(self.nview-1), self.h*self.w, self.h+self.w, 3) / flow_norm_factor -1.0
            n_channel = heatmaps.shape[1]
            heatmaps_sample = heatmaps.view(batch, self.nview, n_channel, self.h, self.w).permute(0,2,1,3,4).contiguous()
            # sample_hm = grid_sample(heatmaps_sample, grid)
            sample_hm = grid_sample(heatmaps_sample, grid, mode='nearest')
            sample_hm_max, max_indice = torch.max(sample_hm, dim=4)
            sample_hm_max = sample_hm_max.view(batch, n_channel, self.nview, self.nview-1, self.h, self.w)
            sample_hm_max = sample_hm_max.permute(0,2,3,1,4,5).contiguous()
            sample_hm_max = sample_hm_max.view(batch*self.nview, self.nview-1, n_channel, self.h, self.w)
            crossview_fused = sample_hm_max

        return crossview_fused

    def roll_on_dim1(self, tensor, offset, maxoffset=None):
        if maxoffset is None:
            maxoffset = self.nview
        offset = offset % maxoffset

        part1 = tensor[:, :offset]
        part2 = tensor[:, offset:]
        res = torch.cat((part2, part1), dim=1).contiguous()
        return res
