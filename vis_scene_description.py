# """
# Quick debug: print per-FRAME Qwen3-VL scene descriptions for ONE scene folder.

# Input: a single scene folder containing per_frame_gt.mp4 (BEV per frame) + CAM_*.mp4, e.g.
#   /data3/.../vis_local/nuscenes_geosplit/gt/scene-0002

# For each frame index it pairs the per-frame BEV (from per_frame_gt.mp4 / per_frame_pred.mp4)
# with that frame's surround cameras, asks Qwen3-VL for a description, and PRINTS it.
# Nothing is saved.


# Env Setup
# -----
# conda create -n qwen3vl python=3.10 -y
# conda activate qwen3vl
# pip install torch torchvision
# pip install "transformers>=4.57.0" accelerate imageio imageio-ffmpeg pillow


# Usage
# -----
# python vis_scene_description.py \
#     --scene_dir vis_local/nuscenes_geosplit/skeptic/scene-0002 \
#     --cam_dir vis_local/nuscenes_geosplit/gt/scene-0002 \
#     --first_frames 2
# """

# import os, glob, argparse
# import imageio.v2 as imageio
# from PIL import Image
# import torch
# from transformers import AutoModelForImageTextToText, AutoProcessor

# # TODO: rethink better text prompts (now why always supported? even with garbage preditions?)
# # TODO: save the results to json or some format
# # TODO: write a standalone eval script for pred vs gt scene descriptions
# SYSTEM = ("You are a map-element auditor for autonomous driving. "
#           "You receive surround-view cameras from an ego vehicle and a BEV map. "
#           "In the BEV, the orange car at center is the ego vehicle; "
#           "green 'B' lines are road boundaries, red 'D' lines are lane dividers, "
#           "blue 'P' lines are pedestrian crossings. "
#           "Coordinates are meters relative to ego (forward=+x, left=+y). "
#           "Ignore dynamic agents (cars, pedestrians) entirely — focus only on static map elements.")
# USER = ("Report the map elements in the BEV using the fixed structure below. "
#         "For each field, use the surround cameras solely to verify whether that element "
#         "is supported or contradicted by real-world visual evidence. "
#         "Be specific and quantitative; avoid vague or narrative language.\n\n"
#         "TOPOLOGY: [straight road / T-intersection / 4-way intersection / merge / roundabout / other]\n"
#         "ROAD_BOUNDARIES: [count, approximate ego-relative position in meters for left and right, "
#         "shape (straight / curving left / curving right / absent on one side)]\n"
#         "LANE_DIVIDERS: [count, spacing in meters, type (dashed / solid), "
#         "do they terminate ahead (yes/no, distance if yes)]\n"
#         "PEDESTRIAN_CROSSINGS: [count, distance ahead in meters, orientation "
#         "(perpendicular / angled), or 'none']\n"
#         "CAMERA_AGREEMENT: [for each element type above, state 'supported', "
#         "'contradicted', or 'uncertain' based on the camera images, with one-phrase reason]")



# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--scene_dir", required=True,
#                    help="folder with per_frame_*.mp4 (BEV); also used for cameras if --cam_dir not set")
#     p.add_argument("--cam_dir", default=None,
#                    help="folder with CAM_*.mp4 (defaults to --scene_dir)")
#     p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
#     p.add_argument("--cam_scale", default=0.5, type=float)
#     p.add_argument("--bev_scale", default=1.0, type=float)
#     p.add_argument("--first_frames", default=None, type=int,
#                    help="process only the first N frames (default: all)")
#     return p.parse_args()


# def read_all_frames(path, scale, max_frames=None):
#     """Return list of PIL RGB frames from an mp4, optionally capped at max_frames."""
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


# def find_bev_mp4(scene_dir):
#     """per_frame_gt.mp4 (GT) or per_frame_pred.mp4 (pred), whichever exists."""
#     for name in ("per_frame_gt.mp4", "per_frame_pred.mp4"):
#         p = os.path.join(scene_dir, name)
#         if os.path.exists(p):
#             return p
#     hits = glob.glob(os.path.join(scene_dir, "per_frame_*.mp4"))
#     return hits[0] if hits else None


# def main():
#     args = parse_args()

#     model = AutoModelForImageTextToText.from_pretrained(
#         args.model, dtype="auto", device_map="auto")
#     processor = AutoProcessor.from_pretrained(args.model)

#     # per-frame BEV from per_frame_gt.mp4 (or per_frame_pred.mp4)
#     bev_mp4 = find_bev_mp4(args.scene_dir)
#     if bev_mp4 is None:
#         raise FileNotFoundError(f"No per_frame_*.mp4 in {args.scene_dir}")
#     bev_frames = read_all_frames(bev_mp4, args.bev_scale, args.first_frames)

#     # read each camera's full frame list
#     cam_root = args.cam_dir if args.cam_dir else args.scene_dir
#     cam_paths = sorted(glob.glob(os.path.join(cam_root, "CAM_*.mp4")))
#     if not cam_paths:
#         raise FileNotFoundError(f"No CAM_*.mp4 in {cam_root}. Use --cam_dir if cameras are elsewhere.")
#     cam_names = [os.path.splitext(os.path.basename(p))[0] for p in cam_paths]
#     cam_frames = [read_all_frames(p, args.cam_scale, args.first_frames) for p in cam_paths]

#     num_frames = min([len(bev_frames)] + [len(f) for f in cam_frames])

#     print(f"Scene: {args.scene_dir}")
#     print(f"BEV:   {os.path.basename(bev_mp4)} ({len(bev_frames)} frames)")
#     print(f"Cameras: {cam_names}")
#     print(f"Frames used: {num_frames}\n")

#     for fi in range(num_frames):
#         content = [{"type": "text", "text": USER},
#                    {"type": "image", "image": bev_frames[fi]},
#                    {"type": "text", "text": "Above: top-down BEV map."}]
#         for cam_name, frames in zip(cam_names, cam_frames):
#             content.append({"type": "image", "image": frames[fi]})
#             content.append({"type": "text", "text": f"Above: camera {cam_name}."})

#         messages = [{"role": "system", "content": SYSTEM},
#                     {"role": "user", "content": content}]
#         inputs = processor.apply_chat_template(
#             messages, tokenize=True, add_generation_prompt=True,
#             return_dict=True, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
#         trimmed = out[0][inputs["input_ids"].shape[1]:]
#         desc = processor.decode(trimmed, skip_special_tokens=True).strip()

#         print(f"===== Frame {fi} =====")
#         print(desc)
#         print()


# if __name__ == "__main__":
#     main()
