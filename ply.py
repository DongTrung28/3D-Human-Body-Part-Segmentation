# paint_lidar_to_ply.py
import os, sys, json, zlib, gc, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", required=True)
    ap.add_argument("--masks-dir", required=True, help="ctx_ts{ts}_cam{cam}_fullpart.png files")
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--use-return", type=int, default=1, choices=[1,2])
    ap.add_argument("--top-only", action="store_true", help="use only TOP lidar if present")
    ap.add_argument("--contexts", default="", help="optional comma separated context filters")
    ap.add_argument("--with-labels", action="store_true", help="add part_id property to PLY")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=0, help="limit for a quick test")
    return ap.parse_args()

def _decode_float_buf(buf, H, W, C):
    arr = np.frombuffer(buf, np.float32)
    if arr.size != H*W*C:
        arr = np.frombuffer(zlib.decompress(buf), np.float32)
    return arr.reshape(H, W, C)

class WaymoContextCache:
    def __init__(self, root: Path):
        self.root = root
        self._cam_cal = {}
        self._lid_cal = {}
        self._tables  = {}

    def cam_calib(self, ctx):
        if ctx not in self._cam_cal:
            self._cam_cal[ctx] = pd.read_parquet(self.root/"camera_calibration"/f"{ctx}.parquet")
        return self._cam_cal[ctx]

    def lid_calib(self, ctx):
        if ctx not in self._lid_cal:
            self._lid_cal[ctx] = pd.read_parquet(self.root/"lidar_calibration"/f"{ctx}.parquet")
        return self._lid_cal[ctx]

    def table(self, name, ctx):
        key = (name, ctx)
        if key not in self._tables:
            self._tables[key] = pd.read_parquet(self.root/name/f"{ctx}.parquet")
        return self._tables[key]

def build_idx_and_xyz_zbuf(cache: WaymoContextCache, ctx: str, ts: int, cam_name: int,
                           use_return=1, top_only=False):
    df_li   = cache.table("lidar", ctx)
    df_prj  = cache.table("lidar_camera_projection", ctx)
    df_lcal = cache.lid_calib(ctx)
    df_ccal = cache.cam_calib(ctx)

    row_cam = df_ccal[df_ccal["key.camera_name"]==cam_name].iloc[0]
    W_img = int(row_cam["[CameraCalibrationComponent].width"])
    H_img = int(row_cam["[CameraCalibrationComponent].height"])
    T_vehicle_cam = np.array(row_cam["[CameraCalibrationComponent].extrinsic.transform"],
                             np.float32).reshape(4,4)

    idx_img = np.full((H_img, W_img), -1, dtype=np.int64)

    tscol, ncol = "key.frame_timestamp_micros", "key.laser_name"
    if use_return == 1:
        ret_val, ret_shape = "[LiDARComponent].range_image_return1.values", "[LiDARComponent].range_image_return1.shape"
        pj_val,  pj_shape  = "[LiDARCameraProjectionComponent].range_image_return1.values", "[LiDARCameraProjectionComponent].range_image_return1.shape"
    else:
        ret_val, ret_shape = "[LiDARComponent].range_image_return2.values", "[LiDARComponent].range_image_return2.shape"
        pj_val,  pj_shape  = "[LiDARCameraProjectionComponent].range_image_return2.values", "[LiDARCameraProjectionComponent].range_image_return2.shape"

    rows_li = df_li[df_li[tscol]==ts]
    if rows_li.empty:
        return idx_img, np.zeros((0,3), np.float32)

    if top_only and (rows_li[ncol]==1).any():
        rows_li = rows_li[rows_li[ncol]==1]

    xyz_list = []
    base = 0
    all_pix, all_li, all_z = [], [], []

    for _, r in rows_li.iterrows():
        laser = int(r[ncol])
        H,W,C = map(int, r[ret_shape])
        ri = _decode_float_buf(r[ret_val], H,W,C)
        rng = ri[...,0].astype(np.float32)

        crow = df_lcal[df_lcal[ncol]==laser].iloc[0]
        vals = crow.get("[LiDARCalibrationComponent].beam_inclination.values", None)
        if vals is not None:
            inc = np.asarray(vals, np.float32).ravel()
            if inc.size == H+1:
                inc = 0.5*(inc[:-1] + inc[1:])
            if inc.size != H:
                inc = np.interp(np.linspace(0, inc.size-1, H, dtype=np.float32),
                                np.arange(inc.size, dtype=np.float32), inc).astype(np.float32)
        else:
            inc = np.linspace(float(crow["[LiDARCalibrationComponent].beam_inclination.min"]),
                              float(crow["[LiDARCalibrationComponent].beam_inclination.max"]),
                              H, dtype=np.float32)

        az = np.linspace(-np.pi, np.pi, W, dtype=np.float32)
        incg = np.repeat(inc[:, None], W, axis=1)
        azg  = np.repeat(az[None, :], H, axis=0)

        x = rng * np.cos(incg) * np.cos(azg)
        y = rng * np.cos(incg) * np.sin(azg)
        z = rng * np.sin(incg)
        xyz_lidar = np.stack([x,y,z], axis=-1).reshape(-1,3)

        T_vehicle_lidar = np.array(crow["[LiDARCalibrationComponent].extrinsic.transform"],
                                   np.float32).reshape(4,4)
        xyz_h = np.c_[xyz_lidar, np.ones((xyz_lidar.shape[0],1), np.float32)]
        xyz_vehicle = (T_vehicle_lidar @ xyz_h.T).T[:, :3].astype(np.float32)
        N = xyz_vehicle.shape[0]
        xyz_list.append(xyz_vehicle)

        prow = df_prj[(df_prj[tscol]==ts) & (df_prj[ncol]==laser)]
        if prow.empty:
            base += N
            continue
        prow = prow.iloc[0]
        PH,PW,PC = map(int, prow[pj_shape])
        P = _decode_float_buf(prow[pj_val], PH,PW,PC)

        cam1 = P[...,0].astype(np.int32); u1 = P[...,1]; v1 = P[...,2]
        cam2 = P[...,3].astype(np.int32); u2 = P[...,4]; v2 = P[...,5]
        m1 = cam1==cam_name
        m2 = cam2==cam_name

        rr1, cc1 = np.where(m1)
        rr2, cc2 = np.where(m2)
        lin1 = rr1*PW + cc1
        lin2 = rr2*PW + cc2

        uu = np.concatenate([u1[m1].astype(np.int32), u2[m2].astype(np.int32)], axis=0)
        vv = np.concatenate([v1[m1].astype(np.int32), v2[m2].astype(np.int32)], axis=0)
        lin = np.concatenate([lin1, lin2], axis=0)

        if uu.size:
            sel_xyz = xyz_vehicle[lin]
            cam_xyz = (T_vehicle_cam @ np.c_[sel_xyz, np.ones((sel_xyz.shape[0],1), np.float32)].T).T[:, :3]
            cam_z   = cam_xyz[:,2]
            ok = (uu>=0)&(uu<W_img)&(vv>=0)&(vv<H_img)&(cam_z>0.1)
            if ok.any():
                pix = vv[ok].astype(np.int64)*W_img + uu[ok].astype(np.int64)
                all_pix.append(pix)
                all_li.append(base + lin[ok])
                all_z.append(cam_z[ok].astype(np.float32))

        del ri, rng, xyz_lidar, xyz_h
        gc.collect()
        base += N

    if not xyz_list:
        return idx_img, np.zeros((0,3), np.float32)

    xyz_all = np.vstack(xyz_list).astype(np.float32)

    if all_pix:
        pix = np.concatenate(all_pix)
        gi  = np.concatenate(all_li)
        z   = np.concatenate(all_z)
        order = np.argsort(z)  # nearest first
        pix = pix[order]; gi = gi[order]
        upix, first = np.unique(pix, return_index=True)
        idx_img_flat = idx_img.ravel()
        idx_img_flat[upix] = gi[first]
        idx_img = idx_img_flat.reshape(H_img, W_img)

    return idx_img, xyz_all

def color_to_part_ids_persistent(palette_json: Path, sampled_bgr):
    if palette_json.is_file():
        with open(palette_json, "r") as f:
            color_to_id = {tuple(map(int,k.split(","))): int(v) for k,v in json.load(f).items()}
    else:
        color_to_id = {}
    next_id = 1 + max(color_to_id.values()) if color_to_id else 1

    ids = np.zeros(sampled_bgr.shape[0], dtype=np.int32)
    for i, c in enumerate(map(tuple, sampled_bgr.astype(np.int32))):
        if c == (0,0,0):
            ids[i] = 0
            continue
        if c not in color_to_id:
            color_to_id[c] = next_id
            next_id += 1
        ids[i] = color_to_id[c]
    return ids, color_to_id

def write_ply_xyzrgb(path, xyz, bgr, part_ids=None):
    xyz = np.asarray(xyz, np.float32)
    bgr = np.asarray(bgr, np.uint8)
    n = xyz.shape[0]
    if part_ids is None:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        rec = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                 ("red","u1"),("green","u1"),("blue","u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        rec["red"], rec["green"], rec["blue"] = bgr[:,2], bgr[:,1], bgr[:,0]  # BGR to RGB
    else:
        part_ids = np.asarray(part_ids, np.int32)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property int part_id\n"
            "end_header\n"
        ).encode("ascii")
        rec = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                 ("red","u1"),("green","u1"),("blue","u1"),
                                 ("part_id","<i4")])
        rec["x"], rec["y"], rec["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        rec["red"], rec["green"], rec["blue"] = bgr[:,2], bgr[:,1], bgr[:,0]
        rec["part_id"] = part_ids

    with open(path, "wb") as f:
        f.write(header)
        rec.tofile(f)
def main():
    import traceback, json
    args = parse_args()
    training_root = Path(args.training_root)
    masks_dir     = Path(args.masks_dir)
    out_ply_dir   = Path(args.out_ply)
    out_ply_dir.mkdir(parents=True, exist_ok=True)
    palette_json  = out_ply_dir / "color_to_part_id.json"
    progress_path = out_ply_dir / "progress.json"
    errors_log    = out_ply_dir / "errors.log"

    # Gather masks
    mask_files = sorted(masks_dir.glob("*_fullpart.png"))
    if args.contexts:
        keep_ctx = set(args.contexts.split(","))
        mask_files = [p for p in mask_files if p.name.split("_ts")[0] in keep_ctx]

    # Resume: figure out which PLYs already exist
    existing = {p.stem for p in out_ply_dir.glob("*.ply")}
    todo = [p for p in mask_files if p.stem not in existing]

    print(f"Found {len(mask_files)} masks, {len(existing)} already done, {len(todo)} to do.")

    cache = WaymoContextCache(training_root)
    t0 = time.time()
    written = 0

    # Persist some progress metadata
    def save_progress(last_name=None, done_increment=False):
        meta = {
            "total_masks": len(mask_files),
            "already_done": len(existing),
            "remaining": len(todo) - written,
            "written_this_run": written,
            "last_processed": last_name,
            "use_return": args.use_return,
            "top_only": args.top_only,
            "ts": time.time()
        }
        with open(progress_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Process in a robust loop
    for i, mpath in enumerate(todo, 1):
        out_stem = mpath.stem + ".ply"
        out_ply  = out_ply_dir / out_stem

        try:
            # Parse ctx/ts/cam just for projection
            name_wo = mpath.stem.replace("_fullpart", "")
            ctx, rest = name_wo.split("_ts", 1)
            ts_str, cam_str = rest.split("_cam", 1)
            ts  = int(ts_str)
            cam = int(cam_str)

            mask_color = cv2.imread(str(mpath), cv2.IMREAD_COLOR)
            if mask_color is None:
                raise RuntimeError(f"read fail {mpath.name}")
            H, W = mask_color.shape[:2]

            idx_img, xyz_vehicle = build_idx_and_xyz_zbuf(
                cache, ctx, ts, cam, args.use_return, args.top_only
            )
            if xyz_vehicle.size == 0 or idx_img.shape != (H, W):
                raise RuntimeError(f"empty xyz or size mismatch {idx_img.shape} vs {(H,W)}")

            N = xyz_vehicle.shape[0]
            rgb_bgr = np.full((N,3), 180, np.uint8)

            ys, xs = np.where(idx_img >= 0)
            li = idx_img[ys, xs]
            sampled = mask_color[ys, xs]
            nonblack = (sampled.sum(axis=1) > 0)
            if nonblack.any():
                rgb_bgr[li[nonblack]] = sampled[nonblack]

            part_ids_full = None
            if args.with_labels and nonblack.any():
                part_sel, cmap = color_to_part_ids_persistent(palette_json, sampled[nonblack])
                part_ids_full = np.zeros(N, dtype=np.int32)
                part_ids_full[li[nonblack]] = part_sel
                if cmap:
                    if palette_json.is_file():
                        with open(palette_json, "r") as f:
                            old = {tuple(map(int,k.split(","))): int(v) for k,v in json.load(f).items()}
                    else:
                        old = {}
                    old.update(cmap)
                    with open(palette_json, "w") as f:
                        json.dump({",".join(map(str,k)): int(v) for k,v in old.items()}, f, indent=2)

            write_ply_xyzrgb(out_ply, xyz_vehicle, rgb_bgr, part_ids_full)
            written += 1


            if (written % args.log_every) == 0:
                dt = time.time() - t0
                print(f"[{written}/{len(todo)}] wrote {out_ply.name} points={N} painted={int(nonblack.sum())}  {dt:.1f}s")
                save_progress(last_name=mpath.name)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress and exiting.")
            save_progress(last_name=mpath.name)
            return
        except Exception as e:
            # Log the error and continue
            msg = f"ERROR on {mpath.name}: {e}\n{traceback.format_exc()}\n"
            sys.stderr.write(msg)
            with open(errors_log, "a", encoding="utf-8") as ef:
                ef.write(msg)
            # also save progress snapshot
            save_progress(last_name=mpath.name)
            continue

    save_progress(last_name="done")
    print(f"Done. Wrote {written} new PLYs in {time.time()-t0:.1f}s. Output: {out_ply_dir}")

if __name__ == "__main__":
    main()