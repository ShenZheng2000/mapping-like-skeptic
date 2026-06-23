import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


@BACKBONES.register_module()
class MobileNetV3(BaseModule):
    """MobileNetV3 backbone built on top of torchvision's implementation.

    Args:
        arch (str): 'large' or 'small'. Defaults to 'large'.
        out_indices (Sequence[int]): Indices into the torchvision
            `features` Sequential whose outputs should be returned.
            For 'large', index 16 is the final 1/32 feature map with
            960 channels. Defaults to (16,).
        frozen_stages (int): Number of leading stages (entries in
            `features`) to freeze. -1 means no freezing.
            Defaults to -1.
        norm_eval (bool): Whether to set BN layers to eval mode during
            training. Defaults to False.
        pretrained (bool): Whether to load torchvision's ImageNet
            pretrained weights. Defaults to True.
    """

    def __init__(self,
                 arch='large',
                 out_indices=(16, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 pretrained=True,
                 init_cfg=None):
        super(MobileNetV3, self).__init__(init_cfg=init_cfg)
        assert arch in ('large', 'small')
        builder = mobilenet_v3_large if arch == 'large' else mobilenet_v3_small
        try:
            # torchvision >= 0.13
            weights = 'IMAGENET1K_V2' if pretrained else None
            net = builder(weights=weights)
        except TypeError:
            # torchvision < 0.13
            net = builder(pretrained=pretrained)

        self.features = net.features
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            m = self.features[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
