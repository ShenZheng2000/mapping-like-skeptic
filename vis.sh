# # nuScenes (predictions, vis_global) - original
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
#     --out_dir vis_global/nuscenes_geosplit/skeptic \
#     --option vis-pred  \
#     --per_frame_result 1

# # nuScenes (predictions, vis_global) - mobilenetv3
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_global/nuscenes_geosplit/skeptic_mobilenetv3 \
#     --option vis-pred \
#     --per_frame_result 1

# # nuScenes (GT, vis_global)
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_global/nuscenes_geosplit/gt  \
#     --option vis-gt \
#     --per_frame_result 0



# # # # nuScenes (predictions, vis_local) - original
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
#     --out_dir vis_local/nuscenes_geosplit/skeptic \
#     --option vis-pred \
#     --orientation bt   # <-- new flag


# # # # # nuScenes (predictions, vis_local) - mobilenetv3
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_local/nuscenes_geosplit/skeptic_mobilenetv3 \
#     --option vis-pred \
#     --orientation bt 


# # # # nuScenes (GT, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_local/nuscenes_geosplit/gt  \
#     --option vis-gt \
#     --orientation bt 




# python tools/tracking/prepare_gt_tracks.py \
#   plugin/configs/skeptic/av2_newsplit/stage3.py  \
#   --out-dir tracking_gts/av2

# Argoverse 2 geosplit (predictions, vis_global) - original
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path work_dirs/stage3/pos_predictions.pkl \
#     --out_dir vis_global/argoverse2_geosplit/skeptic \
#     --option vis-pred  \
#     --per_frame_result 1

# Argoverse 2 geosplit (predictions, vis_global) - mobilenetv3
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py \
#     --data_path work_dirs/stage3_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_global/argoverse2_geosplit/skeptic_stage3_mobilenetv3 \
#     --option vis-pred \
#     --per_frame_result 1

# # Argoverse 2 geosplit (GT, vis_global)
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path datasets/argoverse2_geosplit/av2_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_global/argoverse2_geosplit/gt  \
#     --option vis-gt \
#     --per_frame_result 0


# # Argoverse 2 geosplit (predictions, vis_local) - original
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path work_dirs/stage3/pos_predictions.pkl \
#     --out_dir vis_local/argoverse2_geosplit/skeptic \
#     --option vis-pred \
#     --orientation bt   # <-- new flag

# # Argoverse 2 geosplit (predictions, vis_local) - mobilenetv3
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py \
#     --data_path work_dirs/stage3_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_local/argoverse2_geosplit/skeptic_stage3_mobilenetv3 \
#     --option vis-pred \
#     --orientation bt   # <-- new flag

# # Argoverse 2 geosplit (GT, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path datasets/argoverse2_geosplit/av2_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_local/argoverse2_geosplit/gt \
#     --option vis-gt \
#     --orientation bt   # <-- new flag