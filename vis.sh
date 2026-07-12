############################# nuScenes Experiments #############################
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
python tools/visualization/vis_global.py \
    plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
    --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
    --out_dir vis_global/nuscenes_geosplit/gt  \
    --option vis-gt \
    --per_frame_result 1



# # # # # nuScenes (predictions, vis_local) - original
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
#     --out_dir vis_local/nuscenes_geosplit/skeptic \
#     --option vis-pred \
#     --orientation bt \
#     --boxes_pkl datasets/nuscenes/nuscenes_map_infos_val_with_boxes.pkl

# # # # # # nuScenes (predictions, vis_local) - mobilenetv3
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune_mobilenetv3.py \
#     --data_path work_dirs/mls_nusc_new_3_joint_finetune_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_local/nuscenes_geosplit/skeptic_mobilenetv3 \
#     --option vis-pred \
#     --orientation bt \
#     --boxes_pkl datasets/nuscenes/nuscenes_map_infos_val_with_boxes.pkl

# # # # # nuScenes (GT, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
#     --data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_local/nuscenes_geosplit/gt  \
#     --option vis-gt \
#     --orientation bt \
#     --boxes_pkl datasets/nuscenes/nuscenes_map_infos_val_with_boxes.pkl




############################# Argoverse2 Experiments #############################
# python tools/tracking/prepare_gt_tracks.py \
#   plugin/configs/skeptic/av2_newsplit/stage3.py  \
#   --out-dir tracking_gts/

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
python tools/visualization/vis_global.py \
    plugin/configs/skeptic/av2_newsplit/stage3.py \
    --data_path datasets/argoverse2_geosplit/av2_map_infos_val_gt_tracks.pkl \
    --out_dir vis_global/argoverse2_geosplit/gt  \
    --option vis-gt \
    --per_frame_result 1


# # # Argoverse 2 geosplit (predictions, vis_local) - original
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path work_dirs/stage3/pos_predictions.pkl \
#     --out_dir vis_local/argoverse2_geosplit/skeptic \
#     --option vis-pred \
#     --orientation bt \
#     --boxes_pkl datasets/argoverse2_geosplit/av2_map_infos_val_with_boxes.pkl

# # # Argoverse 2 geosplit (predictions, vis_local) - mobilenetv3
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3_mobilenetv3.py \
#     --data_path work_dirs/stage3_mobilenetv3/pos_predictions.pkl \
#     --out_dir vis_local/argoverse2_geosplit/skeptic_stage3_mobilenetv3 \
#     --option vis-pred \
#     --orientation bt \
#     --boxes_pkl datasets/argoverse2_geosplit/av2_map_infos_val_with_boxes.pkl

# # # Argoverse 2 geosplit (GT, vis_local)
# python tools/visualization/vis_per_frame.py \
#     plugin/configs/skeptic/av2_newsplit/stage3.py \
#     --data_path datasets/argoverse2_geosplit/av2_map_infos_val_gt_tracks.pkl \
#     --out_dir vis_local/argoverse2_geosplit/gt \
#     --option vis-gt \
#     --orientation bt \
#     --boxes_pkl datasets/argoverse2_geosplit/av2_map_infos_val_with_boxes.pkl