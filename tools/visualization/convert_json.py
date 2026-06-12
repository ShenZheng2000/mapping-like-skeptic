import argparse
import mmcv

# maptracker script's cat2id — map by NAME (MapTRv2 type ints are ordered differently)
cat2id = {'ped_crossing': 0, 'divider': 1, 'boundary': 2}


def convert(src, strip_token_prefix=False, rotate=True):
    dst = src.replace('nuscmap_results.json', 'submission_vector.json')
    dst = dst.replace('av2map_results.json', 'submission_vector.json')

    raw = mmcv.load(src)

    out = {}
    for frame in raw['results']:
        token = frame['sample_token']
        # AV2 predictions are keyed 'logid_timestamp'; dataset token is just the timestamp.
        if strip_token_prefix:
            token = str(token).split('_')[-1]
        labels, scores, vectors = [], [], []
        for v in frame['vectors']:
            labels.append(cat2id[v['cls_name']])
            scores.append(float(v['confidence_level']))
            if rotate:
                # nuScenes raw is (x in [-15,15], y in [-30,30]); maptracker wants
                # (x in [-30,30], y in [-15,15]) -> 90deg rotation [x,y]->[y,-x].
                pts = [[p[1], -p[0]] for p in v['pts']]
            else:
                # AV2 raw is ALREADY (x in [-30,30], y in [-15,15]) -> no transform.
                pts = [[p[0], p[1]] for p in v['pts']]
            vectors.append(pts)
        out[token] = {'labels': labels, 'scores': scores, 'vectors': vectors}

    mmcv.dump({'meta': raw.get('meta', {}), 'results': out}, dst)
    print('wrote', dst)
    print('frames:', len(out))
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='path to maptrv2 *map_results.json')
    parser.add_argument('--strip_token_prefix', action='store_true',
                        help='AV2: strip logid_ prefix from sample_token')
    parser.add_argument('--no_rotate', action='store_true',
                        help='AV2: skip the 90deg rotation (AV2 points are already in maptracker convention)')
    args = parser.parse_args()
    convert(args.src, strip_token_prefix=args.strip_token_prefix, rotate=not args.no_rotate)