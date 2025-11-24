
# Data Preparation

This document is adapted from the MapTracker repository. 


## nuScenes
**Step 1.** Download [nuScenes](https://www.nuscenes.org/download) dataset to `./datasets/nuscenes`.


**Step 2.** Generate annotation files for NuScenes dataset

```
python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes --newsplit
```

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/skeptic/nuscenes_newsplit/mls_nusc_new_3_joint_finetune.py  --out-dir tracking_gts/nuscenes --visualize
```

Add the ``--visualize`` flag to visualize the data with element IDs derived from our track generation process, or remove it to save disk memory.  


## Argoverse2

**Step 1.** Download [Argoverse2 (sensor)](https://argoverse.github.io/user-guide/getting_started.html#download-the-datasets) dataset to `./datasets/av2`.

**Step 2.** Generate annotation files for Argoverse2 dataset.

```
python tools/data_converter/argoverse_converter.py --data-root ./datasets/av2 --newsplit
```

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/skeptic/av2_newsplit/mls_av2_new_3_joint_finetune.py  --out-dir tracking_gts/av2 --visualize
```


## Checkpoints

We provide the checkpoints at [this link](https://drive.google.com/drive/folders/1C4SPfUT_kgv5crj0pVJQQ9DfGkuQRlrl?usp=sharing). Please download and place them as ``./work_dirs/pretrained_ckpts``.


## File structures

Make sure the final file structures look like below:

```
mapping-like-skeptic
├── mmdetection3d
├── tools
├── plugin
│   ├── configs
│   ├── models
│   ├── datasets
│   ├── ...
├── work_dirs
│   ├── pretrained_ckpts
│   │   ├── mls_nusc_new_3_joint_finetune
│   │   │   ├── latest.pth
│   │   ├── ...
│   ├── ....
├── datasets
│   ├── nuscenes
│   │   ├── maps <-- used
│   │   ├── samples <-- key frames
│   │   ├── v1.0-test <-- metadata
|   |   ├── v1.0-trainval <-- metadata and annotations
│   │   ├── nuscenes_map_infos_train_newsplit.pkl <-- train annotations
│   │   ├── nuscenes_map_infos_train_newsplit_gt_tracks.pkl <-- train gt tracks
│   │   ├── nuscenes_map_infos_val_newsplit.pkl <-- val annotations
│   │   ├── nuscenes_map_infos_val_newsplit_gt_trakcs.pkl <-- val gt tracks
│   ├── av2
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── maptrv2_val_samples_info.pkl <-- maptr's av2 metadata, used to align the val set
│   │   ├── av2_map_infos_train_newsplit.pkl <-- train annotations
│   │   ├── av2_map_infos_train_newsplit_gt_tracks.pkl <-- train gt tracks
│   │   ├── av2_map_infos_val_newsplit.pkl <-- val annotations
│   │   ├── av2_map_infos_val_newsplit_gt_trakcs.pkl <-- val gt tracks

```
