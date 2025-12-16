
#!/usr/bin/env python3
"""
make_cdgnet_masks.py

Reads a CSV of gait-ready Waymo person rows, pulls the corresponding camera image
from Waymo v2 parquet, optionally crops to the person box, runs a CDGNet predictor,
and saves paired images and masks.

This script avoids heavy dependencies: only numpy, pandas, pyarrow, opencv, and waymo-open-dataset v2.
You must provide a Python function hook that runs your already-working CDGNet and returns
an integer mask of shape (H, W) in image coordinates.

Usage:
  python make_cdgnet_masks.py \
    --data_root /path/to/waymo_v2_parquet \
    --csv /path/to/people_gait_ready_waymo.csv \
    --out_dir /path/to/output_folder \
    --crop person \
    --hook_module cdg_hook.py \
    --hook_fn run_cdgnet

The output folder will contain two subfolders:
  images/  original RGB crops saved as PNG
  masks/   corresponding masks saved as PNG (uint8)

Hook requirements:
  In your hook module file (e.g., cdg_hook.py), implement:

    def run_cdgnet(image_bgr) -> np.ndarray:
        \"\"\"
        Args:
            image_bgr: np.uint8 array of shape (H, W, 3) in BGR order
        Returns:
            mask: np.uint8 array of shape (H, W), 0 = background, >0 = human parts
        \"\"\"
        ...

If you prefer to run the full image (no crop), set --crop none.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from waymo_open_dataset import v2 as wod

# CSV columns we rely on (as seen in your file):
# 'key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name',
# '[CameraBoxComponent].box.center.x', '[CameraBoxComponent].box.center.y',
# '[CameraBoxComponent].box.size.x',   '[CameraBoxComponent].box.size.y'

def load_camera_image_from_parquet(data_root: str, segment: str, frame_ts: int, camera_name: int):
    """
    Read the camera image bytes for the given (segment, timestamp, camera_name).
    Returns decoded BGR image as np.uint8 HxWx3, or None if not found.
    """
    ci_path = os.path.join(data_root, "camera_image", f"{segment}.parquet")
    if not os.path.exists(ci_path):
        raise FileNotFoundError(f"Missing camera_image parquet: {ci_path}")
    ci = pd.read_parquet(ci_path, engine="pyarrow")

    sel = ci[
        (ci["key.segment_context_name"] == segment) &
        (ci["key.frame_timestamp_micros"] == frame_ts) &
        (ci["key.camera_name"] == camera_name)
    ]
    if len(sel) == 0:
        return None

    img_bytes = sel.iloc[0]["[CameraImageComponent].image"]
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def safe_crop(img, cx, cy, w, h):
    H, W = img.shape[:2]
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + int(round(w))
    y1 = y0 + int(round(h))
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None, (0,0,0,0)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hook_module", required=True, help="Path to a .py file that defines your run_cdgnet()")
    ap.add_argument("--hook_fn", default="run_cdgnet", help="Function name in hook module")
    ap.add_argument("--crop", choices=["person","none"], default="person")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for a quick dry run")
    args = ap.parse_args()

    # Load hook
    import importlib.util
    spec = importlib.util.spec_from_file_location("cdg_hook", args.hook_module)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    run_cdgnet = getattr(hook, args.hook_fn)

    # Prepare folders
    out_images = Path(args.out_dir) / "images"
    out_masks  = Path(args.out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = [
        "key.segment_context_name","key.frame_timestamp_micros","key.camera_name",
        "[CameraBoxComponent].box.center.x","[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x","[CameraBoxComponent].box.size.y",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c}")

    rows = df.itertuples(index=False)
    if args.limit > 0:
        rows = list(rows)[:args.limit]

    count = 0
    for r in rows:
        seg   = getattr(r, "key.segment_context_name")
        ts    = int(getattr(r, "key.frame_timestamp_micros"))
        cam   = int(getattr(r, "key.camera_name"))
        cx    = float(getattr(r, "_5"))  # center.x
        cy    = float(getattr(r, "_6"))  # center.y
        bw    = float(getattr(r, "_7"))  # size.x
        bh    = float(getattr(r, "_8"))  # size.y

        img = load_camera_image_from_parquet(args.data_root, seg, ts, cam)
        if img is None:
            print(f"[WARN] image not found for {seg} {ts} cam {cam}")
            continue

        if args.crop == "person":
            crop, xyxy = safe_crop(img, cx, cy, bw, bh)
            if crop is None:
                print(f"[WARN] empty crop for {seg} {ts} cam {cam}")
                continue
            img_for_net = crop
        else:
            img_for_net = img

        # Run your already-installed CDGNet
        try:
            mask = run_cdgnet(img_for_net)
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                raise ValueError("Hook must return a 2D np.uint8 mask")
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] CDGNet hook failed on {seg} {ts} cam {cam}: {e}")
            continue

        # Save image and mask
        stem = f"{seg}_{ts}_{cam}_{count:06d}.png"
        img_out = out_images / stem
        mask_out = out_masks / stem

        # Save the same view that went into the network
        cv2.imwrite(str(img_out), img_for_net)
        cv2.imwrite(str(mask_out), mask)

        count += 1
        if count % 25 == 0:
            print(f"[INFO] processed {count} crops")

    print(f"[DONE] wrote {count} image+mask pairs to {args.out_dir}")

if __name__ == "__main__":
    main()
