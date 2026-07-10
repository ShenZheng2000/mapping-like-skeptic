# Ref: https://github.com/woodfrog/maptracker/issues/33
# Ref: https://github.com/woodfrog/maptracker/issues/51

# MapTRv2 visualization on nuscenes (json -> pkl)
DIR=/data3/shenzhen/Waymo_Projects/MapTR/test/maptrv2_nusc_r50_24ep/Wed_Jun__3_19_08_51_2026/pts_bbox

python tools/visualization/convert_json.py \
  --src $DIR/nuscmap_results.json

python tools/tracking/prepare_pred_tracks.py \
  plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
  --result_path $DIR/submission_vector.json

# NOTE: --per_frame_result 0 to speed up visualizaiton! use 1 for all frames!

python tools/visualization/vis_global.py \
    plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
    --data_path $DIR/pos_predictions_5.pkl \
    --out_dir $DIR/vis_global \
    --option vis-pred \
    --per_frame_result 0

python tools/visualization/vis_per_frame.py \
    plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
    --data_path $DIR/pos_predictions_5.pkl \
    --out_dir $DIR/vis_local \
    --option vis-pred



# MapTRv2 visualization on argoverse 2 (json -> pkl)
DIR=/data3/shenzhen/Waymo_Projects/MapTR/test/maptrv2_av2_3d_r50_6ep_geosplit_allframes/Fri_Jun_12_00_12_54_2026/pts_bbox

python tools/visualization/convert_json.py \
  --src $DIR/av2map_results.json \
  --strip_token_prefix --no_rotate

python tools/tracking/prepare_pred_tracks.py \
  plugin/configs/skeptic/av2_newsplit/stage3.py \
  --result_path $DIR/submission_vector.json

python tools/visualization/vis_global.py \
    plugin/configs/skeptic/av2_newsplit/stage3.py \
    --data_path $DIR/pos_predictions_5.pkl \
    --out_dir $DIR/vis_global \
    --option vis-pred \
    --per_frame_result 0

python tools/visualization/vis_per_frame.py \
    plugin/configs/skeptic/av2_newsplit/stage3.py \
    --data_path $DIR/pos_predictions_5.pkl \
    --out_dir $DIR/vis_local \
    --option vis-pred