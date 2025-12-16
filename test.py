import argparse, os, glob, math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

CAMERA_NAME_MAP = {1:"FRONT",2:"FRONT_LEFT",3:"FRONT_RIGHT",4:"SIDE_LEFT",5:"SIDE_RIGHT"}

def list_contexts(root):
    paths = glob.glob(os.path.join(root, "camera_image", "*.parquet"))
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]

def pq(root, tag, ctx):
    return pd.read_parquet(os.path.join(root, tag, f"{ctx}.parquet"))

def draw_cam_boxes_center(img_bgr, cam_box_df, cam_name, ts):
    sel = cam_box_df[
        (cam_box_df["key.camera_name"] == cam_name) &
        (cam_box_df["key.frame_timestamp_micros"] == ts)
    ]
    canvas = img_bgr.copy()
    for _, b in sel.iterrows():
        cx = float(b["[CameraBoxComponent].box.center.x"])
        cy = float(b["[CameraBoxComponent].box.center.y"])
        w  = float(b["[CameraBoxComponent].box.size.x"])
        h  = float(b["[CameraBoxComponent].box.size.y"])
        x1 = int(round(cx - w/2.0))
        y1 = int(round(cy - h/2.0))
        x2 = int(round(cx + w/2.0))
        y2 = int(round(cy + h/2.0))
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        if "[CameraBoxComponent].type" in b:
            cls = str(b["[CameraBoxComponent].type"])
            cv2.putText(canvas, cls, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return canvas

def box3d_corners(cx, cy, cz, dx, dy, dz, heading):
    sx, sy, sz = dx/2.0, dy/2.0, dz/2.0
    C = np.array([
        [ sx,  sy,  sz], [ sx, -sy,  sz], [-sx, -sy,  sz], [-sx,  sy,  sz],
        [ sx,  sy, -sz], [ sx, -sy, -sz], [-sx, -sy, -sz], [-sx,  sy, -sz],
    ], dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=np.float32)
    return (R @ C.T).T + np.array([cx,cy,cz], dtype=np.float32)

def plot_bev_lidar_boxes(lidar_box_df, ts, out_png):
    sel = lidar_box_df[lidar_box_df["key.frame_timestamp_micros"] == ts]
    fig, ax = plt.subplots(figsize=(6,6))
    for _, bb in sel.iterrows():
        cx = float(bb["[LiDARBoxComponent].box.center.x"])
        cy = float(bb["[LiDARBoxComponent].box.center.y"])
        cz = float(bb["[LiDARBoxComponent].box.center.z"])
        dx = float(bb["[LiDARBoxComponent].box.size.x"])
        dy = float(bb["[LiDARBoxComponent].box.size.y"])
        dz = float(bb["[LiDARBoxComponent].box.size.z"])
        hd = float(bb["[LiDARBoxComponent].box.heading"])
        P = box3d_corners(cx,cy,cz,dx,dy,dz,hd)
        top = np.vstack([P[0:4,:], P[0]])  # footprint
        ax.plot(top[:,0], top[:,1], linewidth=1)
        ax.arrow(cx, cy, 0.5*dx*math.cos(hd), 0.5*dx*math.sin(hd),
                 head_width=0.4, length_includes_head=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Vehicle X [m]"); ax.set_ylabel("Vehicle Y [m]")
    ax.set_title("Waymo v2 BEV 3D boxes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../training folder")
    ap.add_argument("--camera_name", type=int, default=1)
    ap.add_argument("--frame_index", type=int, default=0)
    args = ap.parse_args()

    contexts = list_contexts(args.root)
    if not contexts:
        raise SystemExit("No contexts under camera_image/")
    ctx = contexts[0]
    print("Context:", ctx)

    cam_df       = pq(args.root, "camera_image", ctx)
    cam_box_df   = pq(args.root, "camera_box", ctx)
    lidar_box_df = pq(args.root, "lidar_box", ctx)

    # pick a frame from this camera stream
    stream = cam_df[cam_df["key.camera_name"] == args.camera_name].sort_values("key.frame_timestamp_micros")
    if stream.empty:
        raise SystemExit(f"No frames for camera {args.camera_name} ({CAMERA_NAME_MAP.get(args.camera_name,'?')})")
    row = stream.iloc[min(args.frame_index, len(stream)-1)]
    ts  = int(row["key.frame_timestamp_micros"])
    img_bytes = row["[CameraImageComponent].image"]
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    vis = draw_cam_boxes_center(img, cam_box_df, args.camera_name, ts)
    cv2.imwrite("v2_camera_boxes.png", vis); print("Saved v2_camera_boxes.png")

    plot_bev_lidar_boxes(lidar_box_df, ts, out_png="v2_bev_lidar_boxes.png")

if __name__ == "__main__":
    main()