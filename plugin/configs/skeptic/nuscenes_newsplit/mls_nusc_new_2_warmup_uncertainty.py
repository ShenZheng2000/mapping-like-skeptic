_base_ = ['mls_nusc_new_2_warmup.py']

model = dict(
    head_cfg=dict(
        use_laplace_uncertainty=True,
        loss_reg=dict(type='LaplaceNLLLoss', loss_weight=0.3, _delete_=True),
    )
)
