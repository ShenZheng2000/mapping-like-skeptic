_base_ = [
    './mls_nusc_new_1_bev_pretrain.py'
]

# Add centerline as class index 3 (MapTRv2 does the same: 3 -> 4 classes)
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
    'centerline': 3,
}
num_class = max(list(cat2id.values())) + 1

# Increase queries to handle denser centerline instances
# MapTRv2 scaled num_vec 50->70 (~40%) for the extra class; scale 100->140 here
num_queries = 140

model = dict(
    head_cfg=dict(
        num_queries=num_queries,
        num_classes=num_class,
    ),
    seg_cfg=dict(
        num_classes=num_class,
    ),
)

# Must explicitly override cat2id in all dataset configs — mmcv stores the
# base config's cat2id value at parse time, so the top-level override above
# does NOT propagate into nested data dicts automatically.
data = dict(
    train=dict(cat2id=cat2id),
    val=dict(cat2id=cat2id, eval_config=dict(cat2id=cat2id)),
    test=dict(cat2id=cat2id, eval_config=dict(cat2id=cat2id)),
)
eval_config = dict(cat2id=cat2id)
match_config = dict(cat2id=cat2id)
