_base_ = ['mls_nusc_new_3_joint_finetune.py']

new_train_pkl = '/scratch/shenzhen/Datasets/nuscenes/nuscenes_map_infos_train.pkl'

data = dict(
    test=dict(
        ann_file=new_train_pkl,
        eval_config=dict(ann_file=new_train_pkl),
        test_mode=True,
        seq_split_num=1,
    ),
)
