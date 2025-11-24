"""
    This class is adapted from the BEVFormerBackbone class used in MapTracker
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES
from .prob_proj.grid_mask import GridMask
from mmdet3d.models import builder
from contextlib import nullcontext
import pdb


@BACKBONES.register_module()
class MLSBackbone(nn.Module):
    def __init__(self,
                 roi_size,
                 bev_h,
                 bev_w,
                 img_backbone=None,
                 img_neck=None,
                 use_grid_mask=True,
                 history_steps=None,
                 simple_cnn_predictor=None,
                 **kwargs):
        super(MLSBackbone, self).__init__()

        if simple_cnn_predictor:
            self.simple_cnn_predictor = builder.build_backbone(simple_cnn_predictor)
        else:
            self.simple_cnn_predictor = None

        # image feature
        self.default_ratio = 0.5
        self.default_prob = 0.7
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=self.default_ratio, mode=1,
            prob=self.default_prob)
        self.use_grid_mask = use_grid_mask

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.real_w = roi_size[0]
        self.real_h = roi_size[1]

        self.history_steps = history_steps

        self.init_weights()


    def init_weights(self):
        self.img_backbone.init_weights()
        self.img_neck.init_weights()


    # @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:  # randomly mask some place with a given probability
                img = self.grid_mask(img)

            # resnet
            # it outputs at 3 resolution,
            # (512, 60, 100); (1024, 30, 50); (2048, 15, 25)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            # tuple of length 3 (for each resolution)
            # where each tupl elem is of shape: 6 * (its own dims)
        else:
            return None

        # img_neck is an FPN (Feature Pyramid Network)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward(self, img, img_metas, timestep, history_bev_feats, history_img_metas, all_history_coord,
                all_history_raw_bev_feats, all_history_raw_bev_feats_conf, all_hist_coords,
                *args, prev_bev=None, img_backbone_gradient=True, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        if (self.simple_cnn_predictor is not None and
            not self.simple_cnn_predictor.training and
            img.shape[0] != 1):
            assert False
        scene_name, local_idx = img_metas[0]["scene_name"], img_metas[0]["local_idx"]
        backprop_context = torch.no_grad if img_backbone_gradient is False else nullcontext

        bs, num_cam = img.shape[0:2]
        with backprop_context():
            mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

        if len(history_bev_feats) > 0:
            all_warped_history_feat = []
            all_warped_history_raw_bev_feats = []
            all_warped_history_raw_bev_feats_conf = []
            for b_i in range(bs):
                history_bev_feats_i = torch.stack([feats[b_i] for feats in history_bev_feats], 0)  # t, c, h, w
                history_raw_bev_feats_i = torch.stack([feats[b_i] for feats in all_history_raw_bev_feats], 0)  # T, c, h, w
                history_raw_bev_feats_conf_i = torch.stack([feats[b_i] for feats in all_history_raw_bev_feats_conf], 0).unsqueeze(1)  # T, 1, h, w
                
                _, c, h, w = history_bev_feats_i.shape
                
                if self.training:
                    assert history_bev_feats_i.shape == history_raw_bev_feats_i.shape
                    history_coord = all_history_coord[b_i]
                    history_everything_i = torch.cat([history_bev_feats_i, history_raw_bev_feats_i, history_raw_bev_feats_conf_i], dim=1)  # t, 2c+1, h, w
                    warped_everything_i = F.grid_sample(history_everything_i, history_coord, padding_mode='zeros', align_corners=False)
                    warped_history_feat_i = warped_everything_i[:, :c, :, :]  # t, c, h, w
                    warped_history_raw_bev_feats_i = warped_everything_i[:, c:(2*c), :, :]  # t, c, h, w
                    warped_history_raw_bev_feats_conf_i = warped_everything_i[:, (2*c), :, :].unsqueeze(1)   # t, 1, h, w
                    historical_raw_bev_i = (warped_history_raw_bev_feats_i * warped_history_raw_bev_feats_conf_i).sum(0) / (warped_history_raw_bev_feats_conf_i.sum(0) + 1e-6)  # c, h, w
                    historical_raw_bev_conf_i = ((warped_history_raw_bev_feats_conf_i ** 2).sum(0) / (warped_history_raw_bev_feats_conf_i.sum(0) + 1e-6)).squeeze(0)  # h, w
                else:
                    history_coord = all_history_coord[b_i]  
                    all_histrory_coord = all_hist_coords[b_i]
                    warped_history_feat_i = F.grid_sample(history_bev_feats_i, history_coord, padding_mode='zeros', align_corners=False)  # t, c, h, w
                    
                    raw_hist_feats_n_conf = torch.cat([history_raw_bev_feats_i, history_raw_bev_feats_conf_i], dim=1)  # T, c+1, h, w
                    warped_raw_hist_feats_n_conf = F.grid_sample(raw_hist_feats_n_conf, all_histrory_coord, padding_mode='zeros', align_corners=False)  # T, c+1, h, w
                    assert warped_raw_hist_feats_n_conf.shape[1] == c+1
                    warped_history_raw_bev_feats_i = warped_raw_hist_feats_n_conf[:, :c, :, :]  # T, c, h, w
                    warped_history_raw_bev_feats_conf_i = warped_raw_hist_feats_n_conf[:, c, :, :].unsqueeze(1)  # T, 1, h, w

                    historical_raw_bev_conf_i, max_conf_indices = warped_history_raw_bev_feats_conf_i.squeeze(1).max(dim=0)  # h, w
                    indices_expanded = max_conf_indices.unsqueeze(0).expand(c, h, w)
                    warped_history_raw_bev_feats_i_transposed = warped_history_raw_bev_feats_i.permute(1, 2, 3, 0)  # c, h, w, T
                    historical_raw_bev_i = torch.gather(warped_history_raw_bev_feats_i_transposed, dim=3, index=indices_expanded.unsqueeze(-1)).squeeze(-1) # c, h, w
                
                all_warped_history_feat.append(warped_history_feat_i)
                all_warped_history_raw_bev_feats.append(historical_raw_bev_i)
                all_warped_history_raw_bev_feats_conf.append(historical_raw_bev_conf_i)
            all_warped_history_feat = torch.stack(all_warped_history_feat, dim=0)  # BTCHW
            historical_raw_bev = torch.stack(all_warped_history_raw_bev_feats, dim=0)  # bs, c, h, w
            historical_raw_bev_conf = torch.stack(all_warped_history_raw_bev_feats_conf, dim=0)  # bs, h, w
            prop_bev_feat = all_warped_history_feat[:, -1]
        else:
            all_warped_history_feat = None
            prop_bev_feat = None
            historical_raw_bev = None
            historical_raw_bev_conf = None

        # collect the related mlvl features for each batch
        first_indices = torch.tensor([i for i in range(bs)])
        second_indices = torch.stack([torch.tensor([i for i in range(num_cam)]) for _ in range(bs)])
        related_mlvl_feats = [[] for _ in range(len(mlvl_feats))]  # len(mlvl_feats) = 3 (one for each feature level)
        for i in range(bs):
            for j in range(len(mlvl_feats)):
                related_mlvl_feats[j].append(mlvl_feats[j][first_indices[i], second_indices[i], :, :, :])
        for i in range(len(mlvl_feats)):
            related_mlvl_feats[i] = torch.stack(related_mlvl_feats[i])  # each (bs, 6, 256, ..., ...) - > a tuple of 3 of that

        if num_cam == 6: # nuscenes
            related_cam_names = [["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"] for _ in range(bs)]
        elif num_cam == 7: # argoverse
            related_cam_names = [["ring_front_center", "ring_front_right", "ring_front_left", 
                                    "ring_rear_right", "ring_rear_left", 
                                    "ring_side_right", "ring_side_left"] for _ in range(bs)]
        else:
            assert False

        fov_params = dict(
            index_map=list(),  # to be (bs , 3(cam), h(height_level), h, w, 2)
            cam_mask=list(),   # to be (bs , 3(cam), h, w)
            perspective_distance_mask=list(),  # to be (bs, cam, height_level, 60, 100)
        )
        
        for i in range(bs):
            # pdb.set_trace()
            cur_persp_indices = [img_metas[i]["fov_parameters"]["index_map"][cam_name].to(related_mlvl_feats[0].device)  # related_mlvl_feats[0] for a reference, nothing special
                                    for cam_name in related_cam_names[i]]
            cur_cam_masks = [img_metas[i]["fov_parameters"]["cam_mask"][cam_name].to(related_mlvl_feats[0].device)  # related_mlvl_feats[0] for a reference, nothing special
                                for cam_name in related_cam_names[i]]
            cur_persp_masks = [img_metas[i]["fov_parameters"]["perspective_distance_mask"][cam_name].to(related_mlvl_feats[0].device)
                                for cam_name in related_cam_names[i]] 

            cur_persp_indices = torch.stack(cur_persp_indices)
            cur_cam_masks = torch.stack(cur_cam_masks)
            cur_persp_masks = torch.stack(cur_persp_masks)  # tensor(cam, num_lvl, h, w)

            fov_params["index_map"].append(cur_persp_indices)
            fov_params["cam_mask"].append(cur_cam_masks)
            fov_params["perspective_distance_mask"].append(cur_persp_masks)
        

        bev_feats, merged_raw_bev_feats, merged_raw_conf_level = self.simple_cnn_predictor(
            historical_bev_feats=all_warped_history_feat,  # bs, t, 256, 50, 100
            mlvl_feats=related_mlvl_feats,  # 3 (mlvl) * (bs, num_cam, 256, ..., ...)
            fov_params=fov_params,
            historical_raw_bev=historical_raw_bev,
            historical_raw_bev_conf=historical_raw_bev_conf,
            cam_names=related_cam_names)  # (bs, 3)
        

        return bev_feats, merged_raw_bev_feats, merged_raw_conf_level, mlvl_feats 

