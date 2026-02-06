ulimit -n 65536

# TODO: later: change name to av2_stage1 (or sth like that), to avoid overwriting
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage1.py 4
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage2.py 4
bash ./tools/dist_train.sh plugin/configs/skeptic/av2_newsplit/stage3.py 4 --resume-from work_dirs/stage3/iter_10980.pth