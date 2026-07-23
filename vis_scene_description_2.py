# """
# Print per-frame Qwen3-VL scene descriptions for scene folder(s) using BEV map only.

# For each frame the BEV image is sent to Qwen3-VL, which returns exactly two sentences:
#   1. Map elements present (lane dividers, pedestrian crossings, drivable area boundary).
#   2. Planning instruction derived from those map elements.


# # Env Setup
# # -----
# # conda create -n qwen3vl python=3.10 -y
# # conda activate qwen3vl
# # pip install torch torchvision
# # pip install "transformers>=4.57.0" accelerate imageio imageio-ffmpeg pillow


# Usage
# -----
# # single scene, first 3 frames
# python vis_scene_description_2.py \
#     --one_scene vis_local/nuscenes_geosplit/skeptic/bt_boxes/scene-0002 \
#     --first_frames 3

# # single scene, all frames
# python vis_scene_description_2.py \
#     --one_scene vis_local/nuscenes_geosplit/skeptic/bt_boxes/scene-0002

# # all scenes
# python vis_scene_description_2.py \
#     --all_scenes vis_local/nuscenes_geosplit/skeptic/bt_boxes
# """

# import os, glob, argparse, json
# import imageio.v2 as imageio
# import numpy as np
# from PIL import Image
# import torch
# from transformers import AutoModelForImageTextToText, AutoProcessor

# SYSTEM = ("You are an autonomous driving planner. "
#           "You receive a top-down BEV image centered on the ego vehicle. "
#           "The ego vehicle is driving toward the top of the image.\n"
#           "Map elements: "
#               "green 'B' lines are drivable area boundaries, "
#               "red 'D' lines are lane dividers, "
#               "blue 'P' lines are pedestrian crossings.\n"
#           "Dynamic objects: "
#               "the orange car icon at center is the ego vehicle; "
#               "orange filled boxes are other vehicles, "
#               "cyan filled boxes are pedestrians, "
#               "gray filled boxes are obstacles.")

# USER = ("Respond in exactly two sentences. "
#         "Sentence 1: describe the scene — include map elements (lane dividers, pedestrian "
#         "crossings, drivable area boundaries) and any dynamic objects (vehicles, pedestrians, "
#         "obstacles) with their approximate positions relative to the ego vehicle. "
#         "Sentence 2: give a single concrete planning instruction for the ego vehicle based "
#         "on both the map layout and the dynamic objects present.")


# def parse_args():
#     p = argparse.ArgumentParser()
#     grp = p.add_mutually_exclusive_group(required=True)
#     grp.add_argument("--one_scene",
#                      help="single scene folder with per_frame_*.mp4")
#     grp.add_argument("--all_scenes",
#                      help="parent folder; all immediate subdirs are treated as scenes")
#     p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
#     p.add_argument("--bev_scale", default=1.0, type=float)
#     p.add_argument("--first_frames", default=None, type=int,
#                    help="process only the first N frames per scene (default: all)")
#     return p.parse_args()


# def read_all_frames(path, scale, max_frames=None):
#     reader = imageio.get_reader(path, format="ffmpeg")
#     frames = []
#     for frame in reader:
#         im = Image.fromarray(frame).convert("RGB")
#         if scale != 1.0:
#             im = im.resize((int(im.width * scale), int(im.height * scale)),
#                            Image.Resampling.LANCZOS)
#         frames.append(im)
#         if max_frames is not None and len(frames) >= max_frames:
#             break
#     reader.close()
#     return frames


# def is_blank_frame(img, white_thresh=250, black_thresh=5):
#     arr = np.array(img)
#     mean = arr.mean()
#     return mean >= white_thresh or mean <= black_thresh


# def split_description(desc):
#     parts = desc.split(". ", 1)
#     if len(parts) == 2:
#         return parts[0].rstrip(".") + ".", parts[1]
#     print(f"  [WARN] could not split into two sentences: {desc!r}")
#     return desc, None


# def find_bev_mp4(scene_dir):
#     for name in ("per_frame_gt.mp4", "per_frame_pred.mp4"):
#         p = os.path.join(scene_dir, name)
#         if os.path.exists(p):
#             return p
#     hits = glob.glob(os.path.join(scene_dir, "per_frame_*.mp4"))
#     return hits[0] if hits else None


# def process_scene(scene_dir, model, processor, args):
#     bev_mp4 = find_bev_mp4(scene_dir)
#     if bev_mp4 is None:
#         print(f"[SKIP] no per_frame_*.mp4 in {scene_dir}\n")
#         return

#     bev_frames = read_all_frames(bev_mp4, args.bev_scale, args.first_frames)
#     print(f"Scene: {scene_dir}")
#     print(f"BEV:   {os.path.basename(bev_mp4)} ({len(bev_frames)} frames)\n")

#     results = []
#     for fi, bev_frame in enumerate(bev_frames):
#         if is_blank_frame(bev_frame):
#             print(f"===== Frame {fi} [SKIPPED: blank] =====\n")
#             continue

#         messages = [
#             {"role": "system", "content": SYSTEM},
#             {"role": "user", "content": [
#                 {"type": "image", "image": bev_frame},
#                 {"type": "text", "text": USER},
#             ]},
#         ]
#         inputs = processor.apply_chat_template(
#             messages, tokenize=True, add_generation_prompt=True,
#             return_dict=True, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
#         trimmed = out[0][inputs["input_ids"].shape[1]:]
#         desc = processor.decode(trimmed, skip_special_tokens=True).strip()

#         scene_desc, planning = split_description(desc)
#         results.append({
#             "frame_idx": fi,
#             "scene_description": scene_desc,
#             "planning_instruction": planning,
#         })

#         print(f"===== Frame {fi} =====")
#         print(desc)
#         print()

#     out_path = os.path.join(scene_dir, "scene_descriptions.json")
#     with open(out_path, "w") as f:
#         json.dump({"model": args.model, "frames": results}, f, indent=2)
#     print(f"Saved {len(results)} frames -> {out_path}\n")


# def main():
#     args = parse_args()

#     model = AutoModelForImageTextToText.from_pretrained(
#         args.model, dtype="auto", device_map="auto")
#     processor = AutoProcessor.from_pretrained(args.model)

#     if args.one_scene:
#         scene_dirs = [args.one_scene]
#     else:
#         scene_dirs = sorted(
#             d for d in glob.glob(os.path.join(args.all_scenes, "*"))
#             if os.path.isdir(d)
#         )
#         print(f"Found {len(scene_dirs)} scenes in {args.all_scenes}\n")

#     for scene_dir in scene_dirs:
#         process_scene(scene_dir, model, processor, args)


# if __name__ == "__main__":
#     main()
