"""
Augment an existing nuscenes_map_infos_*.pkl with per-sample dynamic object
annotations (gt_boxes, gt_names, gt_velocity) fetched from the nuScenes SDK.

The existing PKL is not modified; a new file is written alongside it.


Usage
-----
python tools/data_converter/add_boxes_to_infos_nuscenes.py \
    --data-root /scratch/shenzhen/Datasets/nuscenes \
    --pkl      /scratch/shenzhen/Datasets/nuscenes/nuscenes_map_infos_val.pkl \
    --out /scratch/shenzhen/Datasets/nuscenes/nuscenes_map_infos_val_with_boxes.pkl
"""

import argparse
import pickle
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pkl', required=True, help='existing map infos pkl')
    parser.add_argument('--out', required=True, help='output pkl path')
    parser.add_argument('--version', default='v1.0-trainval')
    return parser.parse_args()


def get_boxes_in_ego(nusc, sample_token, e2g_translation, e2g_rotation):
    """Return gt_boxes (N,7) in ego frame, gt_names (N,), gt_velocity (N,2)."""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']

    # get_boxes returns boxes in global frame
    boxes = nusc.get_boxes(lidar_token)
    e2g_q = Quaternion(e2g_rotation)

    gt_boxes, gt_names, gt_velocity = [], [], []
    for box in boxes:
        # in-place global → ego transform (standard nuScenes approach)
        box.translate(-np.array(e2g_translation))
        box.rotate(e2g_q.inverse)

        cx, cy, cz = box.center
        l, w, h = box.wlh[1], box.wlh[0], box.wlh[2]  # wlh: width, length, height
        yaw = box.orientation.yaw_pitch_roll[0]

        gt_boxes.append([cx, cy, cz, l, w, h, yaw])
        gt_names.append(box.name)

        vel_g = nusc.box_velocity(box.token)
        if np.any(np.isnan(vel_g)):
            vel_e = np.array([0.0, 0.0])
        else:
            vel_e = e2g_q.inverse.rotate(vel_g)[:2]
        gt_velocity.append(vel_e)

    return (np.array(gt_boxes, dtype=np.float32).reshape(-1, 7),
            np.array(gt_names),
            np.array(gt_velocity, dtype=np.float32).reshape(-1, 2))


def main():
    args = parse_args()

    print(f'Loading nuScenes SDK from {args.data_root} ...')
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        infos = pickle.load(f)

    print(f'Augmenting {len(infos)} samples ...')
    failed_tokens = []
    kept_infos = []
    for info in infos:
        try:
            gt_boxes, gt_names, gt_velocity = get_boxes_in_ego(
                nusc,
                sample_token=info['token'],
                e2g_translation=info['e2g_translation'],
                e2g_rotation=info['e2g_rotation'],
            )
        except Exception as e:
            failed_tokens.append((info['token'], info.get('scene_name', '?'), str(e)))
            continue
        info['gt_boxes'] = gt_boxes        # (N, 7): x y z l w h yaw in ego
        info['gt_names'] = gt_names        # (N,)  : class string
        info['gt_velocity'] = gt_velocity  # (N, 2): vx vy in ego
        kept_infos.append(info)

    if failed_tokens:
        print(f'\nWARNING: {len(failed_tokens)} sample(s) failed and were excluded:')
        for token, scene, err in failed_tokens:
            print(f'  scene={scene}  token={token}  err={err}')
        print()

    print(f'Kept {len(kept_infos)} / {len(infos)} samples.')
    print(f'Saving to {args.out} ...')
    with open(args.out, 'wb') as f:
        pickle.dump(kept_infos, f)
    print('Done.')


if __name__ == '__main__':
    main()
