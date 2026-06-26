_base_ = [
    'stage3.py'
]

_dim_ = 256


model = dict(
    backbone_cfg=dict(
        img_backbone=dict(
            _delete_=True,
            type='MobileNetV3',
            arch='large',
            out_indices=(6, 12, 16),
            frozen_stages=1,
            norm_eval=True,
            pretrained=False,
            init_cfg=dict(type='Pretrained', checkpoint='ckpts/mobilenet_v3_large-8738ca79.pth'),
        ),
        img_neck=dict(
            _delete_=True,
            type='FPN',
            in_channels=[40, 112, 960],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=3,
            relu_before_extra_convs=True,
        ),
    ),
)

load_from = 'work_dirs/stage2_mobilenetv3/latest.pth'
