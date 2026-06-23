ulimit -n 65536

# TODO: later, if having time, retrain this version (paper reported is 4 GPUs, i think! )
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3.py 4

bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1_mobilenetv3.py 4
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2_mobilenetv3.py 4
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py 4

# bash tools/dist_test.sh  \
#   plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py    \
#   work_dirs/stage3_mobilenetv3/latest.pth  \
#   4 --eval


# python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes --geosplit
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py 8

# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain_mobilenetv3.py 8
# TODO: retrain these with the correct config!
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup_mobilenetv3.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py 8


# bash tools/dist_test.sh  \
#   plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py    \
#   work_dirs/mls_nusc_new_3_joint_finetune_mobilenetv3/latest.pth  \
#   8  --eval

# bash tools/dist_test.sh  \
#   plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup.py    \
#   work_dirs/mls_nusc_new_2_warmup/latest.pth  \
#   8  --eval