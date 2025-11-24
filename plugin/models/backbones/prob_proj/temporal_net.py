import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init
from mmcv.cnn.resnet import conv3x3
from torch import Tensor

from einops import rearrange


class MyResBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 style: str = 'pytorch',
                 with_cp: bool = False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


@NECKS.register_module()
class TemporalNet(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_blocks):
        super(TemporalNet, self).__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        
        layers = []
        
        in_dims = (history_steps+1) * hidden_dims
        self.conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)        

        for _ in range(self.num_blocks):
            layers.append(MyResBlock(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers) 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, history_feats, curr_feat):
        input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
        input_feats = rearrange(input_feats, 'b t c h w -> b (t c) h w') 

        out = self.conv_in(input_feats)
        out = self.bn(out)
        out = self.relu(out)
        out = self.res_layer(out)
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out


@NECKS.register_module()
class ResidualTemporalNet(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_blocks):
        super(ResidualTemporalNet, self).__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks

        layers = []

        in_dims = (history_steps + 1) * hidden_dims
        self.conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)

        for _ in range(self.num_blocks):
            layers.append(MyResBlock(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, history_feats, curr_feat):
        # history feats is the propagated features
        # curr_feat is the new processed feats
        input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
        input_feats = rearrange(input_feats, 'b t c h w -> b (t c) h w')

        out = self.conv_in(input_feats)
        out = self.bn(out)
        # out = out + history_feats.squeeze(1)
        out = self.relu(out)
        out = self.res_layer(out)
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out


@NECKS.register_module()
class MultiHistoryGatedTemporalNet(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_res_blocks, num_gated_blocks, shared_layers):
        # num_gated_blocks, shared_layers are for GatedFusionCNN
        super(MultiHistoryGatedTemporalNet, self).__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.num_gated_blocks = num_gated_blocks

        layers = []

        in_dims = history_steps * hidden_dims
        self.history_conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
        self.gated_fusion_cnn = GatedFusionCNN(hidden_dims, num_gated_blocks, shared_layers=shared_layers)
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)

        for _ in range(self.num_res_blocks):
            layers.append(MyResBlock(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, history_feats, curr_feat):
        # history feats is the historical warped bev feats
        # curr_feat is the new processed feats
        # input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
        history_feats = rearrange(history_feats, 'b t c h w -> b (t c) h w')
        merged_history_feats = self.history_conv_in(history_feats)
        out = self.gated_fusion_cnn(history_feats=merged_history_feats, curr_feats=curr_feat)
        out = self.bn(out)
        # out = out + history_feats.squeeze(1)
        out = self.relu(out)
        out = self.res_layer(out)
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out


@NECKS.register_module()
class GatedTemporalNet(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_res_blocks, num_gated_blocks, shared_layers):
        # num_gated_blocks, shared_layers are for GatedFusionCNN
        super(GatedTemporalNet, self).__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.num_gated_blocks = num_gated_blocks

        layers = []

        in_dims = history_steps * hidden_dims
        self.history_conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
        self.gated_fusion_cnn = GatedFusionCNN(hidden_dims, num_gated_blocks, shared_layers=shared_layers)
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)

        for _ in range(self.num_res_blocks):
            layers.append(MyResBlock(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, history_feats, curr_feat):
        # history feats is the propagated features
        # curr_feat is the new processed feats
        # input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
        history_feats = rearrange(history_feats, 'b t c h w -> b (t c) h w')
        merged_history_feats = self.history_conv_in(history_feats)
        out = self.gated_fusion_cnn(history_feats=merged_history_feats, curr_feats=curr_feat)
        out = self.bn(out)
        # out = out + history_feats.squeeze(1)
        out = self.relu(out)
        out = self.res_layer(out)
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out


class GatedFusionCNN(nn.Module):
    def __init__(self, hidden_dims, num_layers, shared_layers=False):
        super(GatedFusionCNN, self).__init__()
        self.shared_layers = shared_layers
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        # Create a list of GatedFusionCNNCells, one for each layer
        if self.shared_layers:
            self.layers = nn.ModuleList([GatedFusionCNNCell(hidden_dims) for _ in range(1)])
        else:
            self.layers = nn.ModuleList([GatedFusionCNNCell(hidden_dims) for _ in range(num_layers)])

    def forward(self, history_feats, curr_feats):
        for i in range(self.num_layers):
            j = 0 if self.shared_layers else i
            curr_feats = self.layers[j](history_feats, curr_feats)  # Propagate the same history_feats and different curr_feat
        return curr_feats


class GatedFusionCNNCell(nn.Module):
    def __init__(self, hidden_dims):
        super(GatedFusionCNNCell, self).__init__()

        # update gate parameters
        self.W_z = nn.Conv2d(hidden_dims, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.U_z = nn.Conv2d(hidden_dims, 1, kernel_size=3, padding=1, stride=1, bias=True)

        # Reset gate parameters
        self.W_r = nn.Conv2d(hidden_dims, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.U_r = nn.Conv2d(hidden_dims, 1, kernel_size=3, padding=1, stride=1, bias=True)

        # Candidate hidden state parameters
        self.W_h = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1, stride=1, bias=False)
        self.U_h = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1, stride=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))

    def forward(self, history_feats, curr_feat):
        z_t = torch.sigmoid(self.W_z(curr_feat) + self.U_z(history_feats))
        r_t = torch.sigmoid(self.W_r(curr_feat) + self.U_r(history_feats))

        h_tilde = torch.tanh(self.W_h(curr_feat) + self.U_h(history_feats * r_t))

        h_next = z_t * history_feats + (1 - z_t) * h_tilde * self.alpha

        return h_next




