ulimit -n 65536

############################# Argoverse2 Experiments #############################
# python tools/data_converter/argoverse_converter.py --data-root datasets/argoverse2_geosplit

# Preparing 100x50 symlink so to generate different tracking files
# cd /scratch/shenzhen/Datasets/argoverse2_geosplit
# ln -s av2_map_infos_train.pkl av2_map_infos_train_100x50.pkl
# ln -s av2_map_infos_val.pkl av2_map_infos_val_100x50.pkl
# then, proceed with prepare_gt_tracks.py ...

# train
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3.py 4

# train (100m x 50m)
# Maybe: also need to prepare different tracking files for 100m x 50m, if error occurs during training. 
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1_100x50.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2_100x50.py 4
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3_100x50.py 4

# train (mobilenetv3 backbone)
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1_mobilenetv3.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2_mobilenetv3.py 4
# bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py 4

# train (mobilenetv3 backbone; 100m x 50m)


# bash tools/dist_test.sh  \
#   plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py    \
#   work_dirs/stage3_mobilenetv3/latest.pth  \
#   4 --eval



############################# nuScenes Experiments #############################
# NOTE: nuScenes-map-expansion-v1.3.zip and v1.0-trainval-meta.tgz are required, before all the following steps! 
# python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes --geosplit

# Preparing 100x50 symlink so to generate different tracking files
# cd /scratch/shenzhen/Datasets/nuscenes
# ln -s nuscenes_map_infos_train.pkl nuscenes_map_infos_train_100x50.pkl
# ln -s nuscenes_map_infos_val.pkl nuscenes_map_infos_val_100x50.pkl
# then, proceed with prepare_gt_tracks.py ...

# train
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py 8

# train (100m x 50m)
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_1_bev_pretrain.py 8 --resume-from work_dirs/mls_nusc_new_100x50_1_bev_pretrain/iter_34800.pth
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_2_warmup.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_3_joint_finetune.py 8

# train (mobilenetv3 backbone)
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain_mobilenetv3.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup_mobilenetv3.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py 8

# train (mobilenetv3 backbone; 100m x 50m)
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_1_bev_pretrain_mobilenetv3.py 8 --resume-from work_dirs/mls_nusc_new_100x50_1_bev_pretrain_mobilenetv3/iter_6960.pth
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_2_warmup_mobilenetv3.py 8
# bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_100x50_3_joint_finetune_mobilenetv3.py 8


# bash tools/dist_test.sh  \
#   plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py    \
#   work_dirs/mls_nusc_new_3_joint_finetune_mobilenetv3/latest.pth  \
#   8  --eval