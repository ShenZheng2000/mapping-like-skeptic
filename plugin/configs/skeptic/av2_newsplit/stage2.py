_base_ = [
    'mls_av2_new_2_warmup.py'
]

# overwrite PKL paths everywhere
new_val_pkl = '/scratch/shenzhen/datasets/argoverse2_geosplit/av2_map_infos_val.pkl'
new_train_pkl = '/scratch/shenzhen/datasets/argoverse2_geosplit/av2_map_infos_train.pkl'

# dataset overrides for geo-split
data = dict(

    # workers_per_gpu=0, # NOTE: hardcode for now, remove later!!!

    train=dict(
        ann_file=new_train_pkl,
    ),
    val=dict(
        ann_file=new_val_pkl,
        eval_config=dict(
            ann_file=new_val_pkl,   # <--- must override here too
        ),
    ),
    test=dict(
        ann_file=new_val_pkl,
        eval_config=dict(
            ann_file=new_val_pkl,   # <--- must override here too
        ),
    ),
)

# also override top-level eval_config (for safety)
eval_config = dict(
    ann_file=new_val_pkl
)

# and match_config if required
match_config = dict(
    ann_file=new_val_pkl,
)

# checkpoint: load from Stage 1 (geo-split pretrain)
load_from = 'work_dirs/stage1/latest.pth'