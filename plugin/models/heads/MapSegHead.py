import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models import build_loss

from mmdet.models import HEADS

from einops import repeat


@HEADS.register_module(force=True)
class MapSegHead(nn.Module):

    def __init__(self, 
                 num_classes=3,
                 in_channels=256,
                 embed_dims=256,
                 bev_size=(100,50),
                 canvas_size=(200,100),
                 loss_seg=dict(),
                 loss_dice=dict(),
        ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.bev_size = bev_size
        self.canvas_size = canvas_size

        self.loss_seg = build_loss(loss_seg)
        self.loss_dice = build_loss(loss_dice)

        if self.loss_seg.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        assert canvas_size[0] % bev_size[0] == 0, 'canvas size must be a multiple of the bev size'
        self.num_up_blocks = int(np.log2(canvas_size[0] // bev_size[0]))

        self.conv_in = nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv_mid_layers = nn.ModuleList([])
        self.downsample_layers = nn.ModuleList([])
        for _ in range(self.num_up_blocks):
            conv_mid = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv_mid_layers.append(conv_mid)
            self.downsample_layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear'))

        self.conv_out = nn.Conv2d(embed_dims, self.cls_out_channels, kernel_size=1, padding=0)
        

        self.init_weights()
    
    def init_weights(self):
        if self.loss_seg.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            m = self.conv_out
            nn.init.constant_(m.bias, bias_init)
    
    def forward_train(self, bev_features, gts, history_coords):
        # the pred segmentation map is obtained by:
        # applying a 1*1 convolution to the output of conv_mid
        # then the output of conv_mid is passed through a nn.Upsample(0.5) to decrease the spatial dimension
        x = self.relu(self.conv_in(bev_features))  # size stay same (1,256,50,100)
        for conv_mid in self.conv_mid_layers:
            x = conv_mid(x) #   there was 1 layer only, out: (1,256,100,200)
        preds = self.conv_out(x)  # 1*1 conv, the channel number decreases out: (1, 3, 100, 200)

        seg_loss = self.loss_seg(preds, gts)  # mask focal loss
        dice_loss = self.loss_dice(preds, gts)  # mask dice loss
        
        # downsample the features to the original bev size
        seg_feats = x
        for downsample in self.downsample_layers:
            seg_feats = downsample(seg_feats)  # becomes (1 * 256 * 50 * 100)

        return preds, seg_feats, seg_loss, dice_loss
        
    def forward_test(self, bev_features):
        x = self.relu(self.conv_in(bev_features))  # 256, 256; no bias
        for conv_mid in self.conv_mid_layers:
            x = conv_mid(x)     # Upsample x2; 256, 256 (biased); ReLU
        preds = self.conv_out(x)    # 256, 3 (1*1, biased)
        seg_feats = x
        for downsample in self.downsample_layers:
            seg_feats = downsample(seg_feats)
        return preds, seg_feats
    
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
    
    def eval(self):
        super().eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)