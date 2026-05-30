ulimit -n 65536

# TODO: later: change name to av2_stage1 (or sth like that), to avoid overwriting
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3.py 4


# python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes --geosplit
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup.py 8
bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py 8