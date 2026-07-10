"""
Print per-frame Qwen3-VL scene descriptions for ONE scene folder using BEV map only.

Input: a single scene folder containing per_frame_gt.mp4 or per_frame_pred.mp4, e.g.
  /data3/.../vis_local/nuscenes_geosplit/gt/scene-0002

For each frame the BEV image is sent to Qwen3-VL, which returns exactly two sentences:
  1. Map elements present (lane dividers, pedestrian crossings, drivable area boundary).
  2. Planning instruction derived from those map elements.


# Env Setup
# -----
# conda create -n qwen3vl python=3.10 -y
# conda activate qwen3vl
# pip install torch torchvision
# pip install "transformers>=4.57.0" accelerate imageio imageio-ffmpeg pillow


Usage
-----
python vis_scene_description_2.py \
    --scene_dir vis_local/nuscenes_geosplit/gt/scene-0002 \
    --first_frames 2
"""

import os, glob, argparse
import imageio.v2 as imageio
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# TODO: add dynamic objects from gt bboxes, and regenerate video frames. 
# TODO: refine system and user instructions. 
SYSTEM = ("You are an autonomous driving planner. "
          "You receive a top-down BEV map of the area around an ego vehicle. "
          "In the BEV, the orange car at center is the ego vehicle; "
          "green 'B' lines are drivable area boundaries, red 'D' lines are lane dividers, "
          "blue 'P' lines are pedestrian crossings. "
          "Coordinates are meters relative to ego (forward=+x, left=+y).")

USER = ("Respond in exactly two sentences. "
        "Sentence 1: describe the map elements visible in the BEV — include lane dividers, "
        "pedestrian crossings, and drivable area boundaries with their approximate positions. "
        "Sentence 2: give a single concrete planning instruction for the ego vehicle based "
        "solely on those map elements.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True,
                   help="folder with per_frame_*.mp4 (BEV)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--bev_scale", default=1.0, type=float)
    p.add_argument("--first_frames", default=None, type=int,
                   help="process only the first N frames (default: all)")
    return p.parse_args()


def read_all_frames(path, scale, max_frames=None):
    reader = imageio.get_reader(path, format="ffmpeg")
    frames = []
    for frame in reader:
        im = Image.fromarray(frame).convert("RGB")
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)),
                           Image.Resampling.LANCZOS)
        frames.append(im)
        if max_frames is not None and len(frames) >= max_frames:
            break
    reader.close()
    return frames


def find_bev_mp4(scene_dir):
    for name in ("per_frame_gt.mp4", "per_frame_pred.mp4"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(scene_dir, "per_frame_*.mp4"))
    return hits[0] if hits else None


def main():
    args = parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)

    bev_mp4 = find_bev_mp4(args.scene_dir)
    if bev_mp4 is None:
        raise FileNotFoundError(f"No per_frame_*.mp4 in {args.scene_dir}")
    bev_frames = read_all_frames(bev_mp4, args.bev_scale, args.first_frames)

    print(f"Scene: {args.scene_dir}")
    print(f"BEV:   {os.path.basename(bev_mp4)} ({len(bev_frames)} frames)\n")

    for fi, bev_frame in enumerate(bev_frames):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": bev_frame},
                {"type": "text", "text": USER},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        desc = processor.decode(trimmed, skip_special_tokens=True).strip()

        print(f"===== Frame {fi} =====")
        print(desc)
        print()


if __name__ == "__main__":
    main()
