"""
Compute BLEU-4, METEOR, ROUGE-L between GT and predicted scene descriptions.

Each metric is computed separately for scene_description and planning_instruction.

Requirements:
    conda activate qwen3vl
    pip install nltk rouge-score
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

Usage:
    python eval_scene_descriptions.py \
        --gt_dir  vis_local/nuscenes_geosplit/gt/bt_boxes \
        --pred_dirs vis_local/nuscenes_geosplit/skeptic/bt_boxes \
                    vis_local/nuscenes_geosplit/skeptic_mobilenetv3/bt_boxes \
                    /data3/shenzhen/Waymo_Projects/MapTR/test/maptrv2_nusc_r50_24ep/Wed_Jun__3_19_08_51_2026/pts_bbox/vis_local/bt_boxes

"""

import os, glob, json, argparse

import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_pairs(gt_dir, pred_dir, field):
    """
    Walk gt_dir for scene_descriptions.json files, find matching ones in
    pred_dir, align by frame_idx, return (refs, hyps) lists.
    """
    refs, hyps = [], []
    gt_jsons = sorted(glob.glob(os.path.join(gt_dir, "**", "scene_descriptions.json"),
                                recursive=True))
    if not gt_jsons:
        raise FileNotFoundError(f"No scene_descriptions.json found under {gt_dir}")

    matched = skipped = 0
    for gt_path in gt_jsons:
        rel = os.path.relpath(gt_path, gt_dir)          # e.g. scene-0002/scene_descriptions.json
        pred_path = os.path.join(pred_dir, rel)
        if not os.path.exists(pred_path):
            skipped += 1
            continue

        gt_data   = load_json(gt_path)
        pred_data = load_json(pred_path)

        pred_by_fi = {fr["frame_idx"]: fr for fr in pred_data["frames"]}
        for fr in gt_data["frames"]:
            fi = fr["frame_idx"]
            if fi not in pred_by_fi:
                continue
            ref_text  = fr.get(field) or ""
            hyp_text  = pred_by_fi[fi].get(field) or ""
            if ref_text and hyp_text:
                refs.append(ref_text)
                hyps.append(hyp_text)
                matched += 1

    print(f"  scenes matched: {len(gt_jsons)-skipped}/{len(gt_jsons)}, "
          f"frames matched: {matched}, skipped scenes: {skipped}")
    return refs, hyps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(refs, hyps):
    smooth = SmoothingFunction().method1
    ref_tokens  = [[r.lower().split()] for r in refs]
    hyp_tokens  = [h.lower().split()   for h in hyps]

    bleu4 = corpus_bleu(ref_tokens, hyp_tokens,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)

    meteor = float(np.mean([
        meteor_score([r.lower().split()], h.lower().split())
        for r, h in zip(refs, hyps)
    ]))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([
        scorer.score(r, h)["rougeL"].fmeasure
        for r, h in zip(refs, hyps)
    ]))

    return {"BLEU-4": bleu4, "METEOR": meteor, "ROUGE-L": rouge_l}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",    required=True,
                   help="parent folder for GT scenes (contains scene-XXXX/scene_descriptions.json)")
    p.add_argument("--pred_dirs", required=True, nargs="+",
                   help="one or more pred parent folders to compare against GT")
    return p.parse_args()


def main():
    args = parse_args()

    fields = ["scene_description", "planning_instruction"]
    all_results = {}   # pred_dir -> field -> metrics

    for pred_dir in args.pred_dirs:
        # e.g. skeptic/bt_boxes -> "skeptic", skeptic_mobilenetv3/bt_boxes -> "skeptic_mobilenetv3"
        pred_dir = pred_dir.rstrip("/")
        label = os.path.basename(os.path.dirname(pred_dir))
        all_results[label] = {}
        for field in fields:
            print(f"\n[{label}] {field}")
            refs, hyps = collect_pairs(args.gt_dir, pred_dir, field)
            if not refs:
                print("  no aligned frames found, skipping")
                continue
            all_results[label][field] = compute_metrics(refs, hyps)

    # --- print table ---
    col_w = 14
    header_metrics = ["BLEU-4", "METEOR", "ROUGE-L"]
    sep = "-" * (20 + col_w * len(header_metrics))

    for field in fields:
        print(f"\n{'='*60}")
        print(f"Field: {field}")
        print(f"{'='*60}")
        print(f"{'Model':<20}" + "".join(f"{m:>{col_w}}" for m in header_metrics))
        print(sep)
        for label, field_results in all_results.items():
            m = field_results.get(field)
            if m is None:
                print(f"{label:<20}" + "".join(f"{'N/A':>{col_w}}" for _ in header_metrics))
            else:
                print(f"{label:<20}" + "".join(f"{m[k]:>{col_w}.4f}" for k in header_metrics))
        print(sep)


if __name__ == "__main__":
    main()
