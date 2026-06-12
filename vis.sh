# # nuScenes (predictions, vis_global)
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
#     --out_dir vis_global/nuscenes_geosplit/skeptic \
#     --option vis-pred  \
#     --per_frame_result 1

# # nuScenes (GT, vis_global)
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_global/nuscenes_geosplit/gt  \
#     --option vis-gt \
#     --per_frame_result 0

# # Argoverse 2 geosplit (predictions, vis_global)
# python tools/visualization/vis_global.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path work_dirs/stage3/pos_predictions.pkl \
#     --out_dir vis_global/argoverse2_geosplit/skeptic \
#     --option vis-pred  \
#     --per_frame_result 1

python tools/tracking/prepare_gt_tracks.py \
  plugin/configs/skeptic/av2_newsplit/stage3.py  \
  --out-dir tracking_gts/av2


# Argoverse 2 geosplit (GT, vis_global)
python tools/visualization/vis_global.py \
    plugin/configs/skeptic/av2_newsplit/stage3.py \
    --data_path datasets/argoverse2_geosplit/av2_map_infos_val.pkl \
    --out_dir vis_global/argoverse2_geosplit/gt  \
    --option vis-gt \
    --per_frame_result 0



# # # nuScenes (predictions, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
#     --out_dir vis_local/nuscenes_geosplit/skeptic \
#     --option vis-pred

# # # nuScenes (GT, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_local/nuscenes_geosplit/gt  \
#     --option vis-gt