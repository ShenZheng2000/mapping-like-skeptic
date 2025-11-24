# Getting started

This document is adapted from the MapTracker repository. 

We provide the commands for running inference/evaluation, training, and visualization.


## Inference and evaluation


### Inference and evaluate with Chamfer-based mAP


Run the following command to do inference and evaluation using the pretrained checkpoints, assuming 4 GPUs are used.

```
CUDA_VISIBLE_DEVICES=0,1,2,3  bash tools/dist_test.sh  plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py    work_dirs/pretrained_ckpts/mls_nusc_new_3_joint_finetune/latest.pth  4  --eval
```


### Evaluate with C-mAP

Generate prediction matching by
```
python tools/tracking/prepare_pred_tracks.py ${CONFIG} --result_path ${SUBMISSION_FILE} --cons_frames ${COMEBACK_FRAMES}
```

Evaluate with C-mAP by
```
python tools/tracking/calculate_cmap.py ${CONFIG} --result_path ${PRED_MATCHING_INFO}
```

An example evaluation:
```
python tools/tracking/calculate_cmap.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py --result_path ./work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl
```

### Results

By running with the checkpoints we provided in the [data preparation guide](docs/data_preparation.md), the expected results are:

| Dataset     | Range  | Split | Divider | Crossing | Boundary |  mAP  | C-mAP |
|:-----------:|:------:|:-----:|:--------:|:---------:|:---------:|:-----:|:------:|
| nuScenes    | 60×30  |  new  |  36.2    |  49.8     |  50.1     | 45.4  | 37.2  |
| nuScenes    | 100×50 |  new  |  33.1    |  52.5     |  42.9     | 42.8  | 33.5  |
| Argoverse2  | 60×30  |  new  |  76.9    |  72.9     |  67.5     | 72.4  | 62.7  |
| Argoverse2  | 100×50 |  new  |  67.4    |  74.9     |  58.9     | 67.1  | 56.1  |



## Training

The training consists of three stages. We train the models on 4 Nvidia RTX L40S GPUs. 

**Stage 1**: BEV pretraining with semantic segmentation losses:
```
bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_1_bev_pretrain.py 4
```

**Stage 2**: Vector module warmup with a large batch size while freezing the BEV module:
```
bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_2_warmup.py 4
```
Set up the ``load_from=...`` properly in the config file to load the checkpoint from stage 1.

**Stage 3**: Joint finetuning:
```
bash ./tools/dist_train.sh plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py 4
```
Set up the ``load_from=...`` properly in the config file to load the checkpoint from stage 2.



## Visualization

### Global merged reconstruction (merged from local HD maps)

```bash
python tools/visualization/vis_global.py [path to method configuration file under plugin/configs] \
  --data_path [path to the .pkl file] \
  --out_dir [path to the output folder] \
  --option [vis-pred / vis-gt: visualize predicted vectors / visualize ground truth vectors] \
  --per_frame_result 1
```
Set the ``--per_frame_result`` to 1 to generate the per-frame video, the visualization is a bit slow; set it to 0 to only produce the final merged global reconstruction. 


Examples:
```bash
# Visualize prediction
python tools/visualization/vis_global.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
--data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
--out_dir vis_global/nuscenes_new/skeptic \
--option vis-pred  --per_frame_result 1

# Visualize groud truth data
python tools/visualization/vis_global.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
--data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
--out_dir vis_global/nuscenes_new/gt  \
--option vis-gt --per_frame_result 0
```


### Local HD map reconstruction

```bash
python tools/visualization/vis_per_frame.py [path to method configuration file under plugin/configs] \
  --data_path [path to the .pkl file] \
  --out_dir [path to the data folder] \
  --option [vis-pred / vis-gt: visualize predicted vectors / visualize ground truth vectors and input video streams]
```

Note that the input perspective-view videos will be saved when generating the ground truth visualization.


Examples:
```bash
# Visualize prediction
python tools/visualization/vis_per_frame.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
--data_path work_dirs/mls_nusc_new_3_joint_finetune/pos_predictions.pkl \
--out_dir vis_local/nuscenes_new/skeptic \
--option vis-pred

# Visualize groud truth data
python tools/visualization/vis_per_frame.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py \
--data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
--out_dir vis_local/nuscenes_new/gt  \
--option vis-gt
```
