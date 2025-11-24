import torch
import torch.nn as nn
from mmdet3d.models import builder
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class MultiHistorySimpleCNNPredictor(nn.Module):
    def __init__(self,
                 perspective2bev_converter=None,
                 cnn=None,
                 history_steps=4,
                 ignore_prop=False,
                 bev_dims=(50, 100)):
        super(MultiHistorySimpleCNNPredictor, self).__init__()
        self.perspective2bev_converter = builder.build_backbone(perspective2bev_converter)
        self.cnn = builder.build_neck(cnn)
        self.ignore_prop = ignore_prop
        self.bev_h = bev_dims[0]
        self.bev_w = bev_dims[1]
        self.history_steps = history_steps

    def forward(self, historical_bev_feats, mlvl_feats, fov_params, historical_raw_bev, historical_raw_bev_conf, cam_names):
        # historical_bev_feats are already warped, shape: bs, t, 256, 50, 100
        # Convert perspective features to bev features
        # _, _, H, W = prop_bev_feat.shape
        processed_feats, merged_raw_bev_feats, merged_raw_conf_level = self.perspective2bev_converter(mlvl_feats, fov_params, historical_raw_bev, historical_raw_bev_conf)
        if self.ignore_prop:
            return processed_feats, merged_raw_bev_feats, merged_raw_conf_level
        elif historical_bev_feats is None:
            bs, c, h, w = processed_feats.shape
            historical_bev_feats = torch.zeros([bs, self.history_steps, c, h, w]).to(processed_feats.device)
        
        # pdb.set_trace() # check bev feats max min

        # pad the history with 0's if there is not enough number of historical info
        assert len(historical_bev_feats.shape) == 5
        bs, cur_t, c, h, w = historical_bev_feats.shape
        if historical_bev_feats.shape[1] < self.history_steps:
            num_repeat = self.history_steps - cur_t
            zero_bev_feats = torch.zeros([bs, c, h, w]).to(historical_bev_feats.device)
            padding_history_bev_feats = torch.stack([zero_bev_feats, ] * num_repeat, dim=1)
            historical_bev_feats = torch.cat([padding_history_bev_feats, historical_bev_feats], dim=1)

        # process the history to form a single 
        # processed_feats: bs, c, H, W
        # Process through the CNN
        final_feat = self.cnn(historical_bev_feats, processed_feats) # check bev_feats
        # pdb.set_trace()
        assert len(final_feat.shape) == 4

        return final_feat, merged_raw_bev_feats, merged_raw_conf_level

