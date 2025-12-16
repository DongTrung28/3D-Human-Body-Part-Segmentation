import os, glob, argparse
import pandas as pd

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

def list_contexts(training_root):
    patt = os.path.join(training_root, "camera_box", "*.parquet")
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(patt))]

def load_boxes(training_root, context):
    path = os.path.join(training_root, "camera_box", f"{context}.parquet")
    return pd.read_parquet(path)

def normalize_type_is_ped(df):
    col = "[CameraBoxComponent].type"
    if col not in df.columns:
        raise SystemExit("camera_box parquet missing [CameraBoxComponent].type")
    s = df[col]
    if s.dtype == object:
        names = s.astype(str).str.upper()
        return names.isin({"PEDESTRIAN", "TYPE_PEDESTRIAN"})
    return s.astype("int64") == 2   # common enum value for pedestrian

def filter_boxes(df, min_w, min_h, min_area):
    w_col = "[CameraBoxComponent].box.size.x"
    h_col = "[CameraBoxComponent].box.size.y"
    need = ["key.frame_timestamp_micros", "key.camera_name", w_col, h_col]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"camera_box parquet missing column {c}")

    is_ped = normalize_type_is_ped(df)
    w = df[w_col].astype("float32")
    h = df[h_col].astype("float32")
    area = w * h

    keep = is_ped & (w >= min_w) & (h >= min_h) & (area >= min_area)
    return df[keep].copy()

def summarize_kept(df):
    cam_col = "key.camera_name"
    ts_col = "key.frame_timestamp_micros"
    total_boxes = int(len(df))
    # count unique frame camera pairs with at least one kept box
    frame_cam_pairs = df.groupby([ts_col, cam_col], as_index=False).size()
    frames_with_people = int(len(frame_cam_pairs))
    per_cam = df.groupby(cam_col).size().to_dict()
    per_cam_named = {CAMERA_NAMES.get(int(k), str(k)): int(v) for k, v in per_cam.items()}
    return total_boxes, frames_with_people, per_cam_named

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to training folder that contains camera_box")
    ap.add_argument("--min_width",  type=int, default=100,   help="minimum box width in pixels")
    ap.add_argument("--min_height", type=int, default=180,  help="minimum box height in pixels")
    ap.add_argument("--min_area",   type=int, default=25000, help="minimum box area in pixels squared")
    ap.add_argument("--out_csv", default="people_gait_ready_waymo.csv", help="output csv with kept boxes")
    args = ap.parse_args()

    contexts = list_contexts(args.root)
    if not contexts:
        raise SystemExit("no contexts under training/camera_box")

    rows_csv = []
    rows_summary = []
    grand_total = 0
    grand_frames = 0
    grand_per_cam = {name: 0 for name in CAMERA_NAMES.values()}

    for ctx in contexts:
        df = load_boxes(args.root, ctx)
        kept = filter_boxes(df, args.min_width, args.min_height, args.min_area)
        if not kept.empty:
            # write detailed rows to the big csv
            cols_keep = [
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "key.camera_name",
                "[CameraBoxComponent].box.center.x",
                "[CameraBoxComponent].box.center.y",
                "[CameraBoxComponent].box.size.x",
                "[CameraBoxComponent].box.size.y",
                "[CameraBoxComponent].type",
            ]
            cols_keep = [c for c in cols_keep if c in kept.columns]
            kept_subset = kept[cols_keep].copy()
            kept_subset.insert(0, "context", ctx)
            rows_csv.append(kept_subset)

            # per context summary
            total_boxes, frames_with_people, per_cam = summarize_kept(kept)
            rows_summary.append([ctx, total_boxes, frames_with_people,
                                 per_cam.get("FRONT",0),
                                 per_cam.get("FRONT_LEFT",0),
                                 per_cam.get("FRONT_RIGHT",0),
                                 per_cam.get("SIDE_LEFT",0),
                                 per_cam.get("SIDE_RIGHT",0)])
            grand_total += total_boxes
            grand_frames += frames_with_people
            for k, v in per_cam.items():
                grand_per_cam[k] = grand_per_cam.get(k, 0) + v

    if rows_csv:
        big = pd.concat(rows_csv, ignore_index=True)
    else:
        big = pd.DataFrame(columns=["context"])

    big.to_csv(args.out_csv, index=False)

    # print a compact summary
    print(f"contexts scanned {len(contexts)}")
    print(f"kept boxes {grand_total}")
    print(f"frame camera pairs with kept people {grand_frames}")
    print("per camera totals")
    for k in ["FRONT","FRONT_LEFT","FRONT_RIGHT","SIDE_LEFT","SIDE_RIGHT"]:
        print(f"  {k}: {grand_per_cam.get(k,0)}")
    print(f"wrote {args.out_csv}")

if __name__ == "__main__":
    main()
