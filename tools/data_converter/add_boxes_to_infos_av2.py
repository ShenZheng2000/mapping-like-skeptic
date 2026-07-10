"""
Augment an existing av2_map_infos_*.pkl with per-sample dynamic object
annotations (gt_boxes, gt_names) fetched from AV2 annotations.feather files.

The existing PKL is not modified; a new file is written alongside it.

Usage
-----
python tools/data_converter/add_boxes_to_infos_av2.py \
    --data-root /scratch/shenzhen/Datasets/argoverse2_geosplit \
    --split val \
    --pkl /scratch/shenzhen/Datasets/argoverse2_geosplit/av2_map_infos_val.pkl \
    --out /scratch/shenzhen/Datasets/argoverse2_geosplit/av2_map_infos_val_with_boxes.pkl
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='root of argoverse2_geosplit')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--pkl', required=True, help='existing av2 map infos pkl')
    parser.add_argument('--out', required=True, help='output pkl path')
    return parser.parse_args()


def get_timestamp_from_lidar_fpath(lidar_fpath):
    """Extract timestamp_ns from a lidar file path like .../sensors/lidar/315966070559696000.feather"""
    basename = os.path.basename(lidar_fpath)
    return int(os.path.splitext(basename)[0])


def get_boxes_in_ego(ann_df, timestamp_ns):
    """
    Filter annotations for this timestamp and return boxes in ego frame.
    AV2 annotations.feather already stores tx_m/ty_m/tz_m in ego frame.
    Returns gt_boxes (N,7), gt_names (N,).
    """
    frame_anns = ann_df[ann_df['timestamp_ns'] == timestamp_ns]
    if len(frame_anns) == 0:
        return (np.zeros((0, 7), dtype=np.float32),
                np.array([], dtype=object))

    gt_boxes, gt_names = [], []
    for _, row in frame_anns.iterrows():
        cx, cy, cz = row['tx_m'], row['ty_m'], row['tz_m']
        yaw = Quaternion(row['qw'], row['qx'], row['qy'], row['qz']).yaw_pitch_roll[0]
        l, w, h = row['length_m'], row['width_m'], row['height_m']
        gt_boxes.append([cx, cy, cz, l, w, h, yaw])
        gt_names.append(row['category'])

    return (np.array(gt_boxes, dtype=np.float32).reshape(-1, 7),
            np.array(gt_names))


def main():
    args = parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    infos = data['samples']

    # cache annotations per log_id to avoid re-reading the same feather file
    ann_cache = {}
    skipped_logs = set()

    print(f'Augmenting {len(infos)} samples ...')
    kept_infos = []
    for info in infos:
        log_id = info['log_id']

        if log_id not in ann_cache:
            ann_path = os.path.join(args.data_root, args.split, log_id, 'annotations.feather')
            if os.path.exists(ann_path):
                ann_cache[log_id] = pd.read_feather(ann_path)
            else:
                ann_cache[log_id] = None  # no annotations for this log (e.g. test split)

        if ann_cache[log_id] is None:
            skipped_logs.add(log_id)
            continue  # drop this sample entirely

        timestamp_ns = get_timestamp_from_lidar_fpath(info['lidar_fpath'])

        gt_boxes, gt_names = get_boxes_in_ego(
            ann_df=ann_cache[log_id],
            timestamp_ns=timestamp_ns,
        )
        info['gt_boxes'] = gt_boxes   # (N, 7): x y z l w h yaw in ego
        info['gt_names'] = gt_names   # (N,)  : AV2 category string
        kept_infos.append(info)

    if skipped_logs:
        print(f'\nWARNING: {len(skipped_logs)} log(s) had no annotations.feather and were excluded:')
        for log_id in sorted(skipped_logs):
            print(f'  {log_id}')
        print()

    data['samples'] = kept_infos
    print(f'Kept {len(kept_infos)} / {len(infos)} samples.')
    print(f'Saving to {args.out} ...')
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')


if __name__ == '__main__':
    main()
