from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from mmdet.models import BACKBONES
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import math
import pdb


@BACKBONES.register_module()
class FocusedPerspective2BEVConverter(nn.Module):
    def __init__(self,
                 perspective_spatial_res,
                 num_feature_levels,
                 roi_y,
                 num_height_levels,
                 resnet_channels,
                 max_pixel_shift=10,
                 gaussian_window_size=25,
                 ):
        super(FocusedPerspective2BEVConverter, self).__init__()
        self.mask_channels = 32
        assert gaussian_window_size % 2 == 1
        self.max_pixel_shift = max_pixel_shift  # it can go right at most max_pixel_shift / 2
        self.perspective_spatial_res = perspective_spatial_res
        self.num_feature_levels = num_feature_levels
        self.roi_y = roi_y # for normalizing perspective mask
        self.resnet_channels = resnet_channels
        self.gaussian_window_size = gaussian_window_size

        self.bev_fill_pixel = nn.Parameter(torch.randn(resnet_channels))

        # receptive field 11*11
        self.receptive_field_persp_mask_conv = 11
        self.perspective_mask_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.mask_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mask_channels, self.mask_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # receptive field 9*9
        # offset values mu_x, mu_y
        self.receptive_field_offset_predictor = 9
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(resnet_channels + self.mask_channels, resnet_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(resnet_channels + self.mask_channels, resnet_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(resnet_channels + self.mask_channels, 2, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid()
        )

        # receptive field 9*9
        # first dim var_x, second dim var_y, third cov_xy, last dim for the usefullness of the pixel
        self.receptive_field_var_n_conf_predictor = 9
        self.var_n_conf_predictor = nn.Sequential(
            nn.Conv2d(resnet_channels, resnet_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(resnet_channels, resnet_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(resnet_channels, 4, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid()
        )

    
    def predict_offsets(self, perspective_features, perspective_mask_features):
        # bs*num_cam, c, HH (modified HH!!!!), WW both
        assert len(perspective_features.shape) == len(perspective_mask_features.shape)
        assert len(perspective_features.shape) == 4
        x = torch.cat([perspective_features, perspective_mask_features], dim=1)  
        for layer in self.offset_predictor:
            x = layer(x)
            if isinstance(layer, nn.ReLU):  # After every ReLU, re-concatenate mask
                x = torch.cat([x, perspective_mask_features], dim=1)

        HH, WW = self.perspective_spatial_res

        # for mu
        scale_factors = torch.tensor([self.max_pixel_shift / WW, self.max_pixel_shift / HH], device=x.device).view(-1, 1, 1)
        mu = (x[:, 0:2] * 2 - 1) * scale_factors

        return mu

    def predict_var_n_conf(self, perspective_features):
        # bs*num_cam, c, HH (modified HH!!!!), WW
        assert len(perspective_features.shape) == 4
        x = self.var_n_conf_predictor(perspective_features)

        HH, WW = self.perspective_spatial_res

        # for var
        min_var = 0.5 * 3e-4
        max_var = 2e-2
        var_x = (x[:, 0].unsqueeze(1) * (max_var - min_var)) + min_var
        var_y = (x[:, 1].unsqueeze(1) * (max_var - min_var) * ((WW / HH) ** 2)) + (min_var * ((WW / HH) ** 2))

        # for corr_coeff
        corr_coeff = x[:, 2] * 1.9 - 0.95  # dont allow it to be very close to -1 or 1

        # for usefullness mask
        usefullness = x[:, 3]

        # combine
        var_n_conf = torch.cat([var_x, var_y, corr_coeff.unsqueeze(1), usefullness.unsqueeze(1)], dim=1)

        return var_n_conf
    
    def check_topk(self, flat_valid_perspective_indices, topk_weights, topk_x_indices, topk_y_indices, HH, WW):
        if torch.isnan(topk_weights).any() or torch.isinf(topk_weights).any():
            print("NaN or Inf detected in topk_weights")
            torch.save(flat_valid_perspective_indices.cpu(), 'flat_valid_perspective_indices.pt')
            import pdb; pdb.set_trace()

        # Check for NaNs or Infs in topk_x_indices and topk_y_indices
        if (torch.isnan(topk_x_indices).any() or torch.isinf(topk_x_indices).any() or
            torch.isnan(topk_y_indices).any() or torch.isinf(topk_y_indices).any()):
            print("NaN or Inf detected in topk_x_indices or topk_y_indices")
            torch.save(flat_valid_perspective_indices.cpu(), 'flat_valid_perspective_indices.pt')
            import pdb; pdb.set_trace()

        # Check for indices out of bounds
        if (topk_x_indices.min() < 0 or topk_x_indices.max() >= WW or
            topk_y_indices.min() < 0 or topk_y_indices.max() >= HH):
            print("Indices out of bounds detected in topk_x_indices or topk_y_indices")
            torch.save(flat_valid_perspective_indices.cpu(), 'flat_valid_perspective_indices.pt')
            import pdb; pdb.set_trace()

    def get_weights_and_perspective_indices(self, params):
        """
            params: tensor(N, 5) -> parameters of gaussian distribution (mu_x, mu_y, var_x, var_y, corr_coef)
        """
        N = params.shape[0]
        device = params.device

        mean_x = params[:, 0]  # Shape: (N,)
        mean_y = params[:, 1]  # Shape: (N,)
        var_x = params[:, 2]   # Shape: (N,)
        var_y = params[:, 3]   # Shape: (N,)
        cov_xy = params[:, 4] * torch.sqrt(var_x) * torch.sqrt(var_y)  # Shape: (N,)

        # Grid dimensions
        HH, WW = self.perspective_spatial_res
        k = self.gaussian_window_size

        # Convert mean positions from [-1, 1] to grid indices
        x_index = ((mean_x.detach() + 1) / 2) * (WW - 1)  # Shape: (N,)
        y_index = ((mean_y.detach() + 1) / 2) * (HH - 1)  # Shape: (N,)
        x_index = x_index.round()
        y_index = y_index.round()

        # Clamp indices to ensure the window is within bounds
        x_index = x_index.clamp(k // 2, WW - (k // 2 + 1))
        y_index = y_index.clamp(k // 2, HH - (k // 2 + 1))

        # Offsets for the window
        offsets = torch.arange(-(k // 2), (k // 2 + 1), device=device)  # Shape: (k,)
        x_offsets = offsets.view(1, k)
        y_offsets = offsets.view(1, k)

        # Expand x_index and y_index to match offsets
        x_index_expanded = x_index.unsqueeze(-1)  # Shape: (N, 1)
        y_index_expanded = y_index.unsqueeze(-1)  # Shape: (N, 1)

        # Compute the grid indices for the window
        x_indices_window = x_index_expanded + x_offsets  # Shape: (N, k)
        y_indices_window = y_index_expanded + y_offsets  # Shape: (N, k)

        # Create meshgrid for each window
        x_indices_grid = x_indices_window.unsqueeze(-1).expand(-1, -1, k)  # (N, k, k)
        y_indices_grid = y_indices_window.unsqueeze(-2).expand(-1, k, -1)  # (N, k, k)

        # Convert grid indices to normalized coordinates [-1, 1]
        x_coords = (x_indices_grid / (WW - 1)) * 2 - 1  # Shape: (N, k, k)
        y_coords = (y_indices_grid / (HH - 1)) * 2 - 1  # Shape: (N, k, k)
        grid_coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape: (N, k, k, 2)

        # Prepare mean tensor
        mean_x_expanded = mean_x.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1, 1)
        mean_y_expanded = mean_y.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1, 1)
        mu = torch.stack((mean_x_expanded, mean_y_expanded), dim=-1)  # Shape: (N, 1, 1, 2)

        # Compute delta
        delta = grid_coords - mu  # Shape: (N, k, k, 2)
        delta = delta.unsqueeze(-1)  # Shape: (N, k, k, 2, 1)
        delta_T = delta.transpose(-2, -1)  # Shape: (N, k, k, 1, 2)

        # Determinant
        det = var_x * var_y - cov_xy ** 2  # Shape: (N,)

        # Inverse covariance matrices
        inv_cov_matrices = torch.zeros(N, 2, 2, device=device)
        inv_cov_matrices[:, 0, 0] = var_y / det
        inv_cov_matrices[:, 0, 1] = -cov_xy / det
        inv_cov_matrices[:, 1, 0] = -cov_xy / det
        inv_cov_matrices[:, 1, 1] = var_x / det
        inv_cov_matrices = inv_cov_matrices.unsqueeze(1).unsqueeze(1)  # Shape: (N, 1, 1, 2, 2)

        # Mahalanobis distance
        mahal_dist = torch.matmul(delta_T, torch.matmul(inv_cov_matrices, delta))
        mahal_dist = mahal_dist.squeeze(-1).squeeze(-1)  # Shape: (N, k, k)

        # Compute log coefficient
        pi = torch.tensor(math.pi, device=det.device)
        log_coef = -0.5 * torch.log(2 * pi) - 0.5 * torch.log(det)  # Shape: (N,)
        log_coef = log_coef.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1, 1)

        # Compute log of the Gaussian function
        log_f = log_coef - 0.5 * mahal_dist  # Shape: (N, k, k)

        # Flatten the last two dimensions
        log_f_flat = log_f.view(N, -1)  # Shape: (N, k * k)

        # Get the top-k log values and indices
        topk_log_values, topk_indices = torch.topk(log_f_flat, k=k, dim=-1)  # Shape: (N, k)

        # Flatten x_indices_grid and y_indices_grid
        x_indices_flat = x_indices_grid.reshape(N, -1)
        y_indices_flat = y_indices_grid.reshape(N, -1)

        # Gather the top-k grid indices
        topk_x_indices = torch.gather(x_indices_flat, -1, topk_indices)  # Shape: (N, k)
        topk_y_indices = torch.gather(y_indices_flat, -1, topk_indices)  # Shape: (N, k)

        # Compute weights in log-space
        max_log_values = topk_log_values.max(dim=-1, keepdim=True)[0]  # Shape: (N, 1)
        topk_log_weights = topk_log_values - max_log_values  # Shift for numerical stability

        # Convert back to normal space and normalize
        topk_weights = torch.exp(topk_log_weights)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Shape: (N, k)

        return topk_weights, topk_x_indices.long(), topk_y_indices.long()

    def get_query(self, historical_sampled_bev, historical_sampled_bev_conf, reverse_index_map, HH_start_idx):
        # historical_sampled_bev -> bs, 256, 50, 100
        # historical_sampled_bev_conf -> bs, 50, 100
        bs, num_cam, HH, WW, _ = reverse_index_map.shape
        if historical_sampled_bev is None:
            return self.perspective_fill_pixel.view(1, 1, -1, 1, 1).expand(bs, num_cam, -1, (HH - HH_start_idx), WW)

        _, c, h, w = historical_sampled_bev.shape

        cropped_index_map = reverse_index_map.reshape(bs*num_cam, HH, WW, 2)[:, HH_start_idx:, :, :]  # bs*num_cam, HH - HH_start_idx, WW, 2
        full_historical_sampled_bev = historical_sampled_bev + (1 - historical_sampled_bev_conf).unsqueeze(1) * self.perspective_fill_pixel.view(1,256,1,1)  # bs, 256, 50, 100
        full_historical_sampled_bev = full_historical_sampled_bev.unsqueeze(1).expand(-1, num_cam, -1, -1, -1).reshape(bs*num_cam, c, h, w)  #  bs*num_cam, c, h, w
        queries = F.grid_sample(full_historical_sampled_bev, cropped_index_map, mode='bilinear', padding_mode='zeros', align_corners=False) # bs*num_cam, 256, HH - HH_start_idx, WW

        zero_mask = (queries == 0).all(dim=1, keepdim=True)  # Create a mask for completely zero spatial location.
        perspective_fill_pixel_expanded = self.perspective_fill_pixel.view(1, -1, 1, 1).expand_as(queries)
        queries = torch.where(zero_mask, perspective_fill_pixel_expanded, queries)

        return queries.reshape(bs, num_cam, c, (HH - HH_start_idx), WW)


    def forward(self, mlvl_feats, fov_params, historical_sampled_bev=None, historical_sampled_bev_conf=None):
        # historical_sampled_bev -> bs, 256, 50, 100
        # historical_sampled_bev_conf -> bs, 50, 100
        perspective_feats = mlvl_feats[0]  # bs, num_cam, 256, HH, WW; Take the highest res only
        bs, num_cam, c, HH, WW = perspective_feats.shape

        # perspective_feats: tensor(bs, cam, 256, 60, 100)
        # perspective_index_map: tensor(bs, 6(cam), h(height_level), h, w, 2) values between -1 and 1
        perspective_index_map = torch.stack(fov_params["index_map"]).squeeze(2)  # squeeze the height level as it should be already 1
        cam_masks = torch.stack(fov_params["cam_mask"])  # bs, cam, h, w; cam mask doesnt have a height level dimension as it is the mask for the base level 0
        perspective_mask = torch.stack(fov_params["perspective_distance_mask"]).squeeze(2)  # bs, cam, 60, 100
        
        perspective_mask = perspective_mask / self.roi_y # normalize

        # Flatten the batch and camera dimensions: Shape (bs*num_cam, HH, WW)
        flat_perspective_mask = perspective_mask.view(-1, HH, WW) != 0
        row_sums = flat_perspective_mask.sum(dim=0).sum(dim=-1)
        # Find the first row with a non-zero sum
        first_non_zero_row = (row_sums > 0).nonzero(as_tuple=True)[0][0].item()
        HH_start_idx_mc = first_non_zero_row - (self.receptive_field_persp_mask_conv // 2) - 1 if (first_non_zero_row - (self.receptive_field_persp_mask_conv // 2) - 1 >= 0) else 0
        HH_start_idx_op = first_non_zero_row - (self.receptive_field_offset_predictor // 2) - 1 if (first_non_zero_row - (self.receptive_field_offset_predictor // 2) - 1 >= 0) else 0
        HH_start_idx = min(HH_start_idx_mc, HH_start_idx_op)

        # get the mask
        lower_perspective_mask = self.perspective_mask_conv(perspective_mask[..., HH_start_idx:, :].view(bs * num_cam, 1, HH - HH_start_idx, WW)).reshape(bs, num_cam, self.mask_channels, HH - HH_start_idx, WW)

        # use the same first_zero_row here as the sampling from offset preds made through the same bev indices that are used to construct the perspective_mask
        lower_offset_map = self.predict_offsets(perspective_feats[..., HH_start_idx:, :].reshape(bs * num_cam, -1, HH - HH_start_idx, WW), lower_perspective_mask.reshape(bs * num_cam, -1, HH - HH_start_idx, WW))   # bs * num_cam, 2, HH - HH_start_idx, WW 
        offset_map = torch.cat([lower_offset_map.new_zeros((bs * num_cam, 2, HH_start_idx, WW)), lower_offset_map], dim=2)  # bs*num_cam, 2, HH, WW

        # GRID_SAMPLE FROM THE PROCESSED RESULTS THE OFFSET AND STD FOR EACH BEV PIXEL
        # perspective_index_map: tensor(bs, num_cam, h, w, 2) values between -1 and 1
        # offset_map: # bs * num_cam, 2, HH, WW
        _, _, h, w, _ = perspective_index_map.shape
        perspective_index_map = perspective_index_map.reshape(bs * num_cam, h, w, 2)
        mu = F.grid_sample(offset_map, perspective_index_map, mode='bilinear', padding_mode='zeros', align_corners=False).reshape(bs, num_cam, -1, h, w).permute(0, 1, 3, 4, 2)  # gives the offset values bs, num_cam, h, w, 2
        perspective_index_map = perspective_index_map.reshape(bs, num_cam, h, w, 2)  # base persp locations
        mu = mu + perspective_index_map  # bs, num_cam, h, w, 2
        mu_sain = mu.clamp(min=-1, max=1)  # bs, num_cam, h, w, 2

        # according to mu_sain, calculate the new HH_start_idx for var_n_conf operations
        # pdb.set_trace()
        first_non_zero_row = (((mu_sain[...,1].min() + 1) / 2.0) * self.perspective_spatial_res[0]).floor().long()
        HH_start_idx = first_non_zero_row - (self.receptive_field_var_n_conf_predictor // 2) - 1 if (first_non_zero_row - (self.receptive_field_var_n_conf_predictor // 2) - 1 >= 0) else 0
        lower_var_n_conf = self.predict_var_n_conf(perspective_feats[..., HH_start_idx:, :].reshape(bs * num_cam, -1, HH - HH_start_idx, WW))  # bs * num_cam, 4, HH - HH_start_idx2, WW 
        var_n_conf = torch.cat([lower_var_n_conf.new_zeros((bs * num_cam, 4, HH_start_idx, WW)), lower_var_n_conf], dim=2)


        var_n_p = F.grid_sample(var_n_conf[:, 0:3, :, :], mu_sain.reshape(-1, h, w, 2), mode='bilinear', padding_mode='zeros',
                                      align_corners=False).reshape(bs, num_cam, -1, h, w).permute(0, 1, 3, 4, 2)  # gives the variance and corr_coef values bs, num_cam, h, w, 3

        final_perspective_indices = torch.cat([mu_sain, var_n_p], dim=-1)  # bs, num_cam, h, w, 5
        flat_valid_mask = cam_masks.bool().flatten() # bs*num_cam*h*w

        pointers = torch.zeros((bs, num_cam, h, w, 4), dtype=torch.long, device=perspective_feats.device)
        pointers[..., 0] = torch.arange(0, bs, device=perspective_feats.device).view(bs, 1, 1, 1)
        pointers[..., 1] = torch.arange(0, num_cam, device=perspective_feats.device).view(1, num_cam, 1, 1)
        pointers[..., 2] = torch.arange(0, h, device=perspective_feats.device).view(1, 1, h, 1)
        pointers[..., 3] = torch.arange(0, w, device=perspective_feats.device).view(1, 1, 1, w)
        valid_pointers = pointers.reshape(-1, 4)[flat_valid_mask]  # bs * around(6000), 4 = N, 4

        flat_perspective_indices = final_perspective_indices.reshape(-1, 5)  # bs * num_cam * h * w, 5;  num_cam*h*w = 6 * 5000
        flat_valid_perspective_indices = flat_perspective_indices[flat_valid_mask]  # bs * around(6000), 5

        # all: bs * around(6000), k = N, k
        topk_weights, topk_x_indices, topk_y_indices = self.get_weights_and_perspective_indices(flat_valid_perspective_indices)
        self.check_topk(flat_valid_perspective_indices, topk_weights, topk_x_indices, topk_y_indices, HH, WW)
        
        N, k = topk_weights.shape
        assert k == self.gaussian_window_size
        points_to_average = perspective_feats.permute(0, 1, 3, 4, 2)[valid_pointers[:, 0].unsqueeze(-1).expand(-1, k), valid_pointers[:, 1].unsqueeze(-1).expand(-1, k), topk_y_indices, topk_x_indices]  # N, k, 256
        local_weights = var_n_conf[:, 3, :, :].reshape(bs, num_cam, 1, HH, WW).permute(0, 1, 3, 4, 2)[valid_pointers[:, 0].unsqueeze(-1).expand(-1, k), valid_pointers[:, 1].unsqueeze(-1).expand(-1, k), topk_y_indices, topk_x_indices].squeeze(-1)  # N, k
        # didnt want the local weights to sum up to 1 so that if a pixel value is realy uncertain based on topk_indices, let it fade away
        confidence_levels = (topk_weights.detach() * local_weights).sum(dim=1)  # Hopefully of size N :)
        averaged_points = (points_to_average * topk_weights.unsqueeze(-1) * local_weights.unsqueeze(-1)).sum(dim=1)  # N, 256
        
        # split the cameras in a non overlapping manner
        if num_cam == 6: # nuscenes
            grp_1_cam = torch.tensor([0, 4, 5], dtype=valid_pointers.dtype, device=valid_pointers.device)  # front, back left, back right
            grp_2_cam = torch.tensor([1, 2, 3], dtype=valid_pointers.dtype, device=valid_pointers.device)  # front right, front left, back
            cam_grps = [grp_1_cam, grp_2_cam]
        elif num_cam == 7: # argoverse
            grp_1_cam = torch.tensor([0, ], dtype=valid_pointers.dtype, device=valid_pointers.device)  # front_center
            grp_2_cam = torch.tensor([1, 3, 6], dtype=valid_pointers.dtype, device=valid_pointers.device)  # front_right, rear_right, side_left
            grp_3_cam = torch.tensor([2, 4, 5], dtype=valid_pointers.dtype, device=valid_pointers.device)  # front_left, rear_left, side_right
            cam_grps = [grp_1_cam, grp_2_cam, grp_3_cam]
        else:
            assert False
        
        aggregate_grps = torch.zeros(len(cam_grps), bs, h, w, c, device=averaged_points.device, dtype=torch.float32)
        aggregate_conf_levels = torch.zeros(len(cam_grps), bs, h, w, device=averaged_points.device, dtype=torch.float32)
        for i in range(len(cam_grps)):
            grp_i_mask = (valid_pointers[:, 1].unsqueeze(1) == cam_grps[i]).any(dim=1) # N
            batch_indices = valid_pointers[grp_i_mask, 0]  # n1
            h_indices = valid_pointers[grp_i_mask, 2]      # n1
            w_indices = valid_pointers[grp_i_mask, 3]      # n1
            aggregate_grps[i, batch_indices, h_indices, w_indices] = averaged_points[grp_i_mask]
            aggregate_conf_levels[i, batch_indices, h_indices, w_indices] = confidence_levels[grp_i_mask]
        counts = torch.clamp(cam_masks.float().sum(dim=1), min=1)  # bs, h, w
        sampled_bev_feats = aggregate_grps.sum(dim=0) / counts.unsqueeze(-1)  # bs, h, w, c
        sampled_bev_feats_conf_levels = aggregate_conf_levels.sum(dim=0) / counts  # bs, h, w

        # merge with historical bev feats according to the confidence
        if historical_sampled_bev is not None:
            assert historical_sampled_bev_conf is not None
            merged_bev_feats = (sampled_bev_feats * sampled_bev_feats_conf_levels.unsqueeze(-1) + historical_sampled_bev.permute(0,2,3,1) * historical_sampled_bev_conf.unsqueeze(-1)) / (sampled_bev_feats_conf_levels.unsqueeze(-1) + historical_sampled_bev_conf.unsqueeze(-1) + 1e-6)  # bs, h, w, c
            merged_conf_level = (sampled_bev_feats_conf_levels ** 2 + historical_sampled_bev_conf ** 2) / (sampled_bev_feats_conf_levels + historical_sampled_bev_conf + 1e-6)  # bs, h, w
        else:
            merged_bev_feats = sampled_bev_feats
            merged_conf_level = sampled_bev_feats_conf_levels

        # add the bev_fill_pixel as a replacer of uncertain parts
        final_bev_feats = (merged_bev_feats + ((1 - merged_conf_level).unsqueeze(-1) * self.bev_fill_pixel.view(1, 1, 1, 256))).permute(0,3,1,2)  # bs, c, h, w

        return final_bev_feats, merged_bev_feats.permute(0, 3, 1, 2), merged_conf_level